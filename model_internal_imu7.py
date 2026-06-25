"""Internal blocked cross-validation with the seven selected IMU variables.

Selected variables:

    mag_x, mag_y, mag_z,
    rot_x, rot_y, rot_z, rot_w

The script compares three feature sets using exactly the same temporal folds:

    IMU_7
    UWB
    UWB_IMU_7

Scene arguments may be aliases or paths with or without the ``.csv`` suffix,
for example ``data2/scene1`` or ``data3/sceneAB``.
"""

import argparse
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import model_internal as internal


IMU7_COLS = [
    "mag_x",
    "mag_y",
    "mag_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "rot_w",
]

ABLATION_FEATURES = {
    "IMU_7": IMU7_COLS,
    "UWB": internal.base.ANCHOR_COLS,
    "UWB_IMU_7": internal.base.ANCHOR_COLS + IMU7_COLS,
}

DEFAULT_MODELS = ["SVM", "XGBOOST"]
DEFAULT_ABLATIONS = list(ABLATION_FEATURES)
DEFAULT_OUTPUT_DIR = "results_model_internal_imu7"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run blocked internal cross-validation using the seven selected "
            "MAG and rotation variables."
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
        help="Inputs to compare. Default: IMU_7 UWB UWB_IMU_7.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of contiguous folds without shuffle. Default: 5.",
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


def dataset_specific_output_dir(scene_files):
    suffix = "_".join(
        f"{os.path.basename(os.path.dirname(path))}_"
        f"{os.path.splitext(os.path.basename(path))[0]}"
        for path in scene_files
    )
    return f"{DEFAULT_OUTPUT_DIR}_{suffix}"


def validate_features(data):
    missing = [column for column in IMU7_COLS if column not in data.columns]
    if missing:
        raise ValueError(f"Faltan variables necesarias para IMU_7: {missing}")

    invalid = data[IMU7_COLS].isna().sum()
    invalid = invalid[invalid > 0]
    if not invalid.empty:
        raise ValueError(
            "Hay valores ausentes en las variables IMU_7: "
            f"{invalid.to_dict()}"
        )


def save_feature_summary(data, output_dir, scene_specs):
    os.makedirs(output_dir, exist_ok=True)

    summary = (
        data.groupby(internal.base.LABEL_COL)[IMU7_COLS]
        .agg(["count", "mean", "std", "median", "min", "max"])
    )
    summary.to_csv(os.path.join(output_dir, "imu7_statistics_by_label.csv"))

    data[IMU7_COLS].corr(method="pearson").to_csv(
        os.path.join(output_dir, "imu7_correlation_matrix.csv")
    )

    metadata = pd.DataFrame([
        {"parameter": "scenes", "value": ", ".join(scene_specs)},
        {"parameter": "features", "value": ", ".join(IMU7_COLS)},
        {
            "parameter": "description",
            "value": "Raw MAG axes plus raw rotation quaternion components",
        },
    ])
    metadata.to_csv(
        os.path.join(output_dir, "imu7_experiment_metadata.csv"),
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
        internal.resolve_scene_path(scene_spec)
        for scene_spec in args.scenes
    ]
    if args.output_dir is None:
        args.output_dir = dataset_specific_output_dir(selected_files)

    require_uwb = any(
        any(
            column in internal.base.ANCHOR_COLS
            for column in ABLATION_FEATURES[ablation_name]
        )
        for ablation_name in args.ablations
    )
    data = internal.load_selected_scenes(
        selected_files,
        require_uwb=require_uwb,
    )
    validate_features(data)

    label_encoder = LabelEncoder().fit(data[internal.base.LABEL_COL])

    print("Escenas:", args.scenes)
    print("Clases:", [str(label) for label in label_encoder.classes_])
    print("Variables IMU_7:", IMU7_COLS)
    print("Ablaciones:", args.ablations)
    print("Modelos:", args.models)
    print("Épocas:", args.epochs)

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

    save_feature_summary(data, args.output_dir, args.scenes)
    internal.save_outputs(
        results_df=pd.concat(all_results, ignore_index=True),
        global_true=global_true,
        global_pred=global_pred,
        label_encoder=label_encoder,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
