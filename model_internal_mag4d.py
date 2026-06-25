"""Internal blocked cross-validation using a physical MAG 4D representation.

Instead of feeding raw ``mag_x``, ``mag_y`` and ``mag_z`` to the models, this
script rotates the magnetic vector from the watch/device frame to a fixed room
frame using ``rot_x``, ``rot_y``, ``rot_z`` and ``rot_w``. It then derives:

    mag_norm
    mag_heading_sin
    mag_heading_cos
    mag_vertical_ratio

The horizontal heading is measured relative to the bed longitudinal axis,
configured with ``--bed-axis-degrees`` in the room XY plane.

The quaternion convention expected by default is (x, y, z, w) rotating vectors
from device coordinates to room coordinates. Use ``--quaternion-direction
room-to-device`` to apply the inverse rotation if the recorded quaternion uses
the opposite convention.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import model_internal as internal


MAG4D_COLS = [
    "mag_norm",
    "mag_heading_sin",
    "mag_heading_cos",
    "mag_vertical_ratio",
]

ABLATION_FEATURES = {
    "MAG_4D": MAG4D_COLS,
    "UWB": internal.base.ANCHOR_COLS,
    "UWB_MAG_4D": internal.base.ANCHOR_COLS + MAG4D_COLS,
}

DEFAULT_MODELS = ["SVM", "XGBOOST"]
DEFAULT_ABLATIONS = list(ABLATION_FEATURES)
DEFAULT_OUTPUT_DIR = "results_model_internal_mag4d"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run blocked internal cross-validation using the physical MAG 4D "
            "representation instead of raw magnetometer axes."
        )
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=internal.DEFAULT_SCENES,
        help=(
            "Scene aliases or CSV paths, with or without .csv. Examples: "
            "data2/scene1 and data3/sceneAB."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "Models: SVM, XGBOOST, SIMPLE_LSTM, STACKED_LSTM or CNN_2LSTM. "
            "Default: SVM XGBOOST."
        ),
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        type=str.upper,
        choices=tuple(ABLATION_FEATURES),
        default=DEFAULT_ABLATIONS,
        help="Inputs to compare. Default: MAG_4D UWB UWB_MAG_4D.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of contiguous folds without shuffle. Default: 5.",
    )
    parser.add_argument(
        "--bed-axis-degrees",
        type=float,
        default=0.0,
        help=(
            "Direction of the bed longitudinal axis in the room XY frame, "
            "measured counter-clockwise from room +X. Default: 0 degrees."
        ),
    )
    parser.add_argument(
        "--quaternion-direction",
        choices=["device-to-room", "room-to-device"],
        default="device-to-room",
        help=(
            "Direction represented by rot_x/y/z/w. Default: device-to-room."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. By default, a dataset-specific directory is "
            f"created from {DEFAULT_OUTPUT_DIR}."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=internal.base.EPOCHS,
        help="Epochs for neural models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=internal.base.BATCH_SIZE,
        help="Batch size for neural models.",
    )
    return parser.parse_args()


def rotate_vectors_by_quaternion(vectors, quaternions, inverse=False):
    """Rotate Nx3 vectors using normalized quaternions stored as Nx4 x,y,z,w."""
    vectors = np.asarray(vectors, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)

    norms = np.linalg.norm(quaternions, axis=1)
    invalid = (~np.isfinite(norms)) | (norms < 0.5)
    if invalid.any():
        raise ValueError(
            f"Hay {int(invalid.sum())} cuaterniones inválidos o de norma < 0.5."
        )

    q = quaternions / norms[:, None]
    q_xyz = q[:, :3].copy()
    q_w = q[:, 3:4]
    if inverse:
        q_xyz *= -1.0

    # Efficient quaternion-vector rotation:
    # v' = v + 2*w*(q_xyz x v) + 2*(q_xyz x (q_xyz x v))
    first_cross = np.cross(q_xyz, vectors)
    rotated = (
        vectors
        + 2.0 * q_w * first_cross
        + 2.0 * np.cross(q_xyz, first_cross)
    )
    return rotated


def add_mag4d_features(
    data,
    bed_axis_degrees=0.0,
    quaternion_direction="device-to-room",
):
    """Add room-frame physical MAG features to a processed scene dataframe."""
    required = [
        "mag_x",
        "mag_y",
        "mag_z",
        "rot_x",
        "rot_y",
        "rot_z",
        "rot_w",
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Faltan columnas necesarias para MAG 4D: {missing}")

    output = data.copy()
    magnetic_device = output[["mag_x", "mag_y", "mag_z"]].to_numpy(
        dtype=np.float64
    )
    quaternion_cols = ["rot_x", "rot_y", "rot_z", "rot_w"]
    quaternion_frame = output[quaternion_cols].astype(np.float64).copy()
    quaternion_norms = np.linalg.norm(quaternion_frame.to_numpy(), axis=1)
    invalid_quaternions = (~np.isfinite(quaternion_norms)) | (
        quaternion_norms < 0.5
    )
    if invalid_quaternions.any():
        print(
            "AVISO: se reemplazan "
            f"{int(invalid_quaternions.sum())} cuaterniones con norma < 0.5 "
            "usando el cuaternión válido temporalmente más cercano."
        )
        quaternion_frame.loc[invalid_quaternions, quaternion_cols] = np.nan
        quaternion_frame = quaternion_frame.ffill().bfill()
    quaternions = quaternion_frame.to_numpy(dtype=np.float64)
    magnetic_room = rotate_vectors_by_quaternion(
        magnetic_device,
        quaternions,
        inverse=(quaternion_direction == "room-to-device"),
    )

    theta = np.deg2rad(bed_axis_degrees)
    bed_axis = np.array([np.cos(theta), np.sin(theta), 0.0])
    cross_axis = np.array([-np.sin(theta), np.cos(theta), 0.0])

    bed_component = magnetic_room @ bed_axis
    cross_component = magnetic_room @ cross_axis
    vertical_component = magnetic_room[:, 2]

    magnitude = np.linalg.norm(magnetic_room, axis=1)
    horizontal_magnitude = np.hypot(bed_component, cross_component)
    safe_magnitude = np.where(magnitude > 1e-12, magnitude, 1.0)
    safe_horizontal = np.where(
        horizontal_magnitude > 1e-12,
        horizontal_magnitude,
        1.0,
    )

    output["mag_norm"] = magnitude
    output["mag_heading_sin"] = cross_component / safe_horizontal
    output["mag_heading_cos"] = bed_component / safe_horizontal
    output["mag_vertical_ratio"] = vertical_component / safe_magnitude

    zero_horizontal = horizontal_magnitude <= 1e-12
    if zero_horizontal.any():
        output.loc[
            zero_horizontal,
            ["mag_heading_sin", "mag_heading_cos"],
        ] = 0.0

    if not np.isfinite(output[MAG4D_COLS].to_numpy()).all():
        raise ValueError("La transformación MAG 4D ha producido valores no finitos.")

    return output


def save_feature_summary(data, output_dir, args):
    os.makedirs(output_dir, exist_ok=True)
    summary = (
        data.groupby(internal.base.LABEL_COL)[MAG4D_COLS]
        .agg(["count", "mean", "std", "median", "min", "max"])
    )
    summary.to_csv(os.path.join(output_dir, "mag4d_statistics_by_label.csv"))

    data[MAG4D_COLS].corr().to_csv(
        os.path.join(output_dir, "mag4d_correlation_matrix.csv")
    )

    metadata = pd.DataFrame([
        {"parameter": "bed_axis_degrees", "value": args.bed_axis_degrees},
        {
            "parameter": "quaternion_direction",
            "value": args.quaternion_direction,
        },
        {"parameter": "features", "value": ", ".join(MAG4D_COLS)},
        {"parameter": "scenes", "value": ", ".join(args.scenes)},
    ])
    metadata.to_csv(
        os.path.join(output_dir, "mag4d_transformation_metadata.csv"),
        index=False,
    )


def main():
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds debe ser al menos 2.")

    internal.base.MODELS_TO_RUN = args.models
    internal.base.EPOCHS = args.epochs
    internal.base.BATCH_SIZE = args.batch_size
    internal.base.check_dependencies()

    selected_files = [
        internal.resolve_scene_path(scene_spec) for scene_spec in args.scenes
    ]
    if args.output_dir is None:
        scene_suffix = "_".join(
            f"{os.path.basename(os.path.dirname(path))}_"
            f"{os.path.splitext(os.path.basename(path))[0]}"
            for path in selected_files
        )
        args.output_dir = f"{DEFAULT_OUTPUT_DIR}_{scene_suffix}"
    require_uwb = any(
        any(column in internal.base.ANCHOR_COLS for column in ABLATION_FEATURES[name])
        for name in args.ablations
    )
    data = internal.load_selected_scenes(
        selected_files,
        require_uwb=require_uwb,
    )
    data = add_mag4d_features(
        data,
        bed_axis_degrees=args.bed_axis_degrees,
        quaternion_direction=args.quaternion_direction,
    )

    label_encoder = LabelEncoder().fit(data[internal.base.LABEL_COL])

    print("Escenas:", args.scenes)
    print("Clases:", [str(label) for label in label_encoder.classes_])
    print("Ablaciones:", args.ablations)
    print(f"Eje longitudinal camas: {args.bed_axis_degrees:g} grados")
    print("Dirección del cuaternión:", args.quaternion_direction)
    print("Modelos:", args.models)

    all_results = []
    global_true = {}
    global_pred = {}

    for ablation_name in args.ablations:
        feature_cols = ABLATION_FEATURES[ablation_name]
        print("\n" + "#" * 70)
        print(f"Ablación: {ablation_name} | columnas: {feature_cols}")
        print("#" * 70)

        X, y = internal.make_internal_windows(
            data=data,
            label_encoder=label_encoder,
            feature_cols=feature_cols,
        )
        results, y_true, y_pred = internal.run_internal_cv(
            X=X,
            y=y,
            models=args.models,
            folds=args.folds,
            label_encoder=label_encoder,
            ablation_name=ablation_name,
            allow_missing_train_classes=True,
        )
        all_results.append(results)
        for model_name in args.models:
            key = (ablation_name, model_name)
            global_true[key] = y_true[model_name]
            global_pred[key] = y_pred[model_name]

    save_feature_summary(data, args.output_dir, args)
    internal.save_outputs(
        results_df=pd.concat(all_results, ignore_index=True),
        global_true=global_true,
        global_pred=global_pred,
        label_encoder=label_encoder,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
