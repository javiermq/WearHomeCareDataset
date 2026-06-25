"""Cross-scene validation with physical MAG 4D and UWB ablations.

This is the MAG 4D counterpart of
``model_cnn2lstm_xgb_svm_5scenes.py``. Each fold leaves one complete scene out
for testing and trains on all remaining scenes.

Compared feature sets:

    MAG_4D:
        mag_norm, mag_heading_sin, mag_heading_cos, mag_vertical_ratio

    UWB:
        distance_00_01, distance_00_02

    UWB_MAG_4D:
        distance_00_01, distance_00_02 plus the four MAG 4D features

MAG 4D is calculated by rotating the device magnetometer vector to a fixed
room frame with rot_x/y/z/w. The horizontal heading is expressed relative to
the configured bed longitudinal axis.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

import model_cnn2lstm_xgb_svm_5scenes as base
from model_internal_mag4d import (
    ABLATION_FEATURES,
    MAG4D_COLS,
    add_mag4d_features,
)


SCENE_FILES = {
    f"scene{i}": f"data/scene{i}.csv"
    for i in range(1, 6)
}

DEFAULT_SCENES = [
    os.path.splitext(os.path.basename(path))[0]
    for path in base.SCENE_FILES
]
DEFAULT_MODELS = ["SVM", "XGBOOST"]
DEFAULT_ABLATIONS = list(ABLATION_FEATURES)
DEFAULT_OUTPUT_DIR = "results_cross_scene_models_mag4d"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Leave-one-scene-out validation comparing MAG 4D, UWB and "
            "UWB+MAG 4D."
        )
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=tuple(SCENE_FILES),
        default=DEFAULT_SCENES,
        help=(
            "Scenes participating in cross-scene validation. Default matches "
            "the base script: scene1 scene2 scene3 scene4."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "Models: SIMPLE_LSTM, STACKED_LSTM, CNN_2LSTM, XGBOOST or SVM. "
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
        "--bed-axis-degrees",
        type=float,
        default=0.0,
        help=(
            "Direction of the bed longitudinal axis in room XY coordinates, "
            "counter-clockwise from room +X. Default: 0."
        ),
    )
    parser.add_argument(
        "--quaternion-direction",
        choices=["device-to-room", "room-to-device"],
        default="device-to-room",
        help="Direction represented by rot_x/y/z/w. Default: device-to-room.",
    )
    parser.add_argument(
        "--require-real-uwb",
        action="store_true",
        help=(
            "For UWB configurations, require at least one real reading from "
            "each anchor inside every window. By default, use the temporally "
            "filled UWB values produced by the base preprocessing."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=base.EPOCHS,
        help="Epochs for neural models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=base.BATCH_SIZE,
        help="Batch size for neural models.",
    )
    return parser.parse_args()


def load_scenes(args):
    scenes = []
    for scene_name in args.scenes:
        data = base.load_scene_long_format(SCENE_FILES[scene_name])
        data = add_mag4d_features(
            data,
            bed_axis_degrees=args.bed_axis_degrees,
            quaternion_direction=args.quaternion_direction,
        )
        scenes.append((scene_name, data))
        print(
            f"{scene_name}: {len(data)} segundos; clases: "
            f"{data[base.LABEL_COL].value_counts().sort_index().to_dict()}; "
            f"UWB real: "
            f"{dict((column, int(data[f'has_{column}'].sum())) for column in base.ANCHOR_COLS)}"
        )
    return scenes


def make_scene_windows(
    scene_data,
    label_encoder,
    feature_cols,
    require_real_uwb,
):
    X_raw = scene_data[feature_cols].to_numpy(dtype=np.float32)
    y_raw = label_encoder.transform(scene_data[base.LABEL_COL])

    uses_uwb = any(column in base.ANCHOR_COLS for column in feature_cols)
    anchor_present = None
    if uses_uwb and require_real_uwb:
        anchor_present = scene_data[base.ANCHOR_PRESENT_COLS].to_numpy(
            dtype=np.int32
        )

    X, y, discarded, total, discard_pct = base.make_centered_windows(
        X_raw,
        y_raw,
        anchor_present=anchor_present,
        past=base.PAST_SECONDS,
        future=base.FUTURE_SECONDS,
    )
    return X, y, {
        "candidate_windows": total,
        "selected_windows": len(y),
        "discarded_windows": discarded,
        "discarded_pct": discard_pct,
    }


def scale_windows(X_train, X_test):
    scaler = StandardScaler()
    n_features = X_train.shape[2]
    X_train_scaled = scaler.fit_transform(
        X_train.reshape(-1, n_features)
    ).reshape(X_train.shape)
    X_test_scaled = scaler.transform(
        X_test.reshape(-1, n_features)
    ).reshape(X_test.shape)
    return X_train_scaled, X_test_scaled


def run_cross_scene(scenes, label_encoder, args):
    n_classes = len(label_encoder.classes_)
    fold_rows = []
    selection_rows = []
    global_true = {}
    global_pred = {}

    for ablation_name in args.ablations:
        feature_cols = ABLATION_FEATURES[ablation_name]
        windows_by_scene = {}

        print("\n" + "#" * 70)
        print(f"Ablación: {ablation_name} | columnas: {feature_cols}")
        print("#" * 70)

        for scene_name, scene_data in scenes:
            X, y, stats = make_scene_windows(
                scene_data=scene_data,
                label_encoder=label_encoder,
                feature_cols=feature_cols,
                require_real_uwb=args.require_real_uwb,
            )
            windows_by_scene[scene_name] = (X, y)
            decoded = (
                label_encoder.inverse_transform(y)
                if len(y) else np.asarray([], dtype=str)
            )
            class_counts = (
                pd.Series(decoded).value_counts().sort_index().to_dict()
                if len(decoded) else {}
            )
            selection_rows.append({
                "ablation": ablation_name,
                "scene": scene_name,
                **stats,
                "class_counts": str(class_counts),
            })
            print(
                f"{scene_name}: {len(y)}/{stats['candidate_windows']} ventanas; "
                f"clases: {class_counts}"
            )

        for model_name in args.models:
            global_true[(ablation_name, model_name)] = []
            global_pred[(ablation_name, model_name)] = []

        for test_scene_name, _ in scenes:
            X_test, y_test = windows_by_scene[test_scene_name]
            train_parts = [
                windows_by_scene[scene_name]
                for scene_name, _ in scenes
                if scene_name != test_scene_name
            ]
            nonempty_train = [
                (X, y) for X, y in train_parts
                if len(y) > 0
            ]

            if len(y_test) == 0 or not nonempty_train:
                print(
                    f"Saltando test={test_scene_name}: no hay ventanas "
                    f"suficientes para {ablation_name}."
                )
                continue

            X_train = np.concatenate([X for X, _ in nonempty_train])
            y_train = np.concatenate([y for _, y in nonempty_train])
            X_train, X_test_scaled = scale_windows(X_train, X_test)

            train_classes = label_encoder.inverse_transform(
                np.unique(y_train)
            ).tolist()
            test_classes = label_encoder.inverse_transform(
                np.unique(y_test)
            ).tolist()

            print("\n" + "=" * 70)
            print(
                f"{ablation_name} | entrenar con todas menos "
                f"{test_scene_name} | test={test_scene_name}"
            )
            print(
                f"Ventanas train={len(y_train)}, test={len(y_test)} | "
                f"clases train={train_classes}, test={test_classes}"
            )
            print("=" * 70)

            for model_name in args.models:
                result = base.train_predict_model(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test_scaled,
                    y_test=y_test,
                    n_classes=n_classes,
                )
                y_pred = result["y_pred"]
                key = (ablation_name, model_name)
                global_true[key].extend(y_test.tolist())
                global_pred[key].extend(y_pred.tolist())
                fold_rows.append({
                    "ablation": ablation_name,
                    "model": model_name,
                    "test_scene": test_scene_name,
                    "test_loss": result["test_loss"],
                    "test_acc": result["test_acc"],
                    "n_train_samples": len(y_train),
                    "n_test_samples": len(y_test),
                    "train_classes": ",".join(map(str, train_classes)),
                    "test_classes": ",".join(map(str, test_classes)),
                })
                print(
                    f"{model_name}: loss={result['test_loss']:.4f}, "
                    f"acc={result['test_acc']:.4f}"
                )

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(selection_rows),
        global_true,
        global_pred,
    )


def save_outputs(
    results,
    selection,
    global_true,
    global_pred,
    label_encoder,
    args,
):
    os.makedirs(args.output_dir, exist_ok=True)

    results.to_csv(
        os.path.join(args.output_dir, "results_cross_scene_mag4d_all.csv"),
        index=False,
    )
    selection.to_csv(
        os.path.join(args.output_dir, "windows_cross_scene_mag4d.csv"),
        index=False,
    )

    summary = (
        results.groupby(["ablation", "model"], as_index=False)
        .agg(
            mean_test_acc=("test_acc", "mean"),
            std_test_acc=("test_acc", "std"),
            mean_test_loss=("test_loss", "mean"),
            std_test_loss=("test_loss", "std"),
            tested_scenes=("test_scene", "count"),
        )
        .sort_values("mean_test_acc", ascending=False)
    )
    summary.to_csv(
        os.path.join(args.output_dir, "summary_cross_scene_mag4d.csv"),
        index=False,
    )

    metadata = pd.DataFrame([
        {"parameter": "scenes", "value": ", ".join(args.scenes)},
        {"parameter": "ablations", "value": ", ".join(args.ablations)},
        {"parameter": "bed_axis_degrees", "value": args.bed_axis_degrees},
        {
            "parameter": "quaternion_direction",
            "value": args.quaternion_direction,
        },
        {"parameter": "require_real_uwb", "value": args.require_real_uwb},
        {"parameter": "mag4d_features", "value": ", ".join(MAG4D_COLS)},
    ])
    metadata.to_csv(
        os.path.join(args.output_dir, "cross_scene_mag4d_metadata.csv"),
        index=False,
    )

    labels = np.arange(len(label_encoder.classes_))
    class_names = [str(label) for label in label_encoder.classes_]

    for (ablation_name, model_name), true_values in sorted(global_true.items()):
        if not true_values:
            continue
        y_true = np.asarray(true_values, dtype=np.int32)
        y_pred = np.asarray(
            global_pred[(ablation_name, model_name)],
            dtype=np.int32,
        )
        safe_name = f"{ablation_name.lower()}_{model_name.lower()}"

        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        )
        pd.DataFrame(report).transpose().to_csv(
            os.path.join(
                args.output_dir,
                f"classification_report_{safe_name}.csv",
            )
        )

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        pd.DataFrame(
            cm,
            index=[f"Real {name}" for name in class_names],
            columns=[f"Pred {name}" for name in class_names],
        ).to_csv(
            os.path.join(
                args.output_dir,
                f"confusion_matrix_{safe_name}.csv",
            )
        )

    print(f"\nResultados guardados en: {args.output_dir}")
    print(summary)


def main():
    args = parse_args()
    if len(args.scenes) < 2:
        raise ValueError("Cross-scene requiere al menos dos escenas.")

    base.MODELS_TO_RUN = args.models
    base.EPOCHS = args.epochs
    base.BATCH_SIZE = args.batch_size
    base.check_dependencies()

    scenes = load_scenes(args)
    labels = pd.concat(
        [data[base.LABEL_COL] for _, data in scenes],
        ignore_index=True,
    )
    label_encoder = LabelEncoder().fit(labels)

    print("Escenas:", args.scenes)
    print("Clases:", [str(label) for label in label_encoder.classes_])
    print("Ablaciones:", args.ablations)
    print("Modelos:", args.models)
    print(f"Eje longitudinal camas: {args.bed_axis_degrees:g} grados")
    print("Dirección cuaternión:", args.quaternion_direction)
    print("Exigir UWB real por ventana:", args.require_real_uwb)

    results, selection, global_true, global_pred = run_cross_scene(
        scenes,
        label_encoder,
        args,
    )
    if results.empty:
        raise ValueError("No se pudo evaluar ningún fold cross-scene.")

    save_outputs(
        results,
        selection,
        global_true,
        global_pred,
        label_encoder,
        args,
    )


if __name__ == "__main__":
    main()
