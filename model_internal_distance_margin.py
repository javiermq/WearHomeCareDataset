"""Classify samples where both UWB distances are similar.

This experiment keeps only windows whose central instant satisfies:

    abs(distance_00_01 - distance_00_02) <= --distance-threshold

It then compares UWB-only against UWB plus magnetometer (compass) using the
same blocked cross-validation and model implementations as model_internal.py.
Distances and the threshold are expressed in centimetres.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import model_internal as internal


FEATURE_SETS = {
    "UWB": internal.base.ANCHOR_COLS,
    "UWB_BRUJULA": internal.base.ANCHOR_COLS + ["mag_x", "mag_y", "mag_z"],
}

DEFAULT_MODELS = ["SVM", "XGBOOST"]
DEFAULT_OUTPUT_DIR = "results_model_internal_distance_margin"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare UWB and UWB+compass only where both UWB distances "
            "differ by at most the selected margin."
        )
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        required=True,
        help=(
            "Maximum absolute difference between the two UWB distances, "
            "in centimetres."
        ),
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of contiguous CV folds. Default: 5.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "Models to run: SVM, XGBOOST, SIMPLE_LSTM, STACKED_LSTM "
            "or CNN_2LSTM. Default: SVM XGBOOST."
        ),
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=tuple(internal.DATA2_SCENE_FILES),
        default=list(internal.DATA2_SCENE_FILES),
        help="Scenes to load. Default: scene1 scene2.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        type=str.upper,
        choices=tuple(FEATURE_SETS),
        default=list(FEATURE_SETS),
        help="Feature sets to compare. Default: UWB UWB_BRUJULA.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}.",
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


def make_margin_windows(data, label_encoder, feature_cols, threshold):
    """Create scene-safe windows and retain only UWB-ambiguous centres."""
    all_windows = []
    all_labels = []
    selection_rows = []

    past = internal.base.PAST_SECONDS
    future = internal.base.FUTURE_SECONDS

    for scene_name, scene_data in data.groupby("scene", sort=False):
        scene_data = scene_data.reset_index(drop=True)
        X_raw = scene_data[feature_cols].to_numpy(dtype=np.float32)
        y_raw = label_encoder.transform(scene_data[internal.base.LABEL_COL])
        d1 = scene_data["distance_00_01"].to_numpy(dtype=np.float32)
        d2 = scene_data["distance_00_02"].to_numpy(dtype=np.float32)

        scene_windows = []
        scene_labels = []
        selected_differences = []

        for center in range(past, len(scene_data) - future):
            difference = float(abs(d1[center] - d2[center]))
            if difference > threshold:
                continue

            start = center - past
            end = center + future + 1
            scene_windows.append(X_raw[start:end])
            scene_labels.append(y_raw[center])
            selected_differences.append(difference)

        if scene_windows:
            all_windows.append(np.asarray(scene_windows, dtype=np.float32))
            all_labels.append(np.asarray(scene_labels, dtype=np.int32))

        label_counts = {}
        if scene_labels:
            decoded = label_encoder.inverse_transform(
                np.asarray(scene_labels, dtype=np.int32)
            )
            label_counts = pd.Series(decoded).value_counts().sort_index().to_dict()

        selection_rows.append({
            "scene": scene_name,
            "distance_threshold_cm": threshold,
            "candidate_windows": max(0, len(scene_data) - past - future),
            "selected_windows": len(scene_labels),
            "selected_pct": (
                100.0 * len(scene_labels) / max(1, len(scene_data) - past - future)
            ),
            "mean_distance_difference_cm": (
                float(np.mean(selected_differences))
                if selected_differences else np.nan
            ),
            "max_distance_difference_cm": (
                float(np.max(selected_differences))
                if selected_differences else np.nan
            ),
            "class_counts": str(label_counts),
        })

        print(
            f"{scene_name}: {len(scene_labels)} ventanas con "
            f"|d1-d2| <= {threshold:g} cm; clases: {label_counts}"
        )

    if not all_windows:
        raise ValueError(
            "El margen no selecciona ninguna ventana. "
            "Prueba un --distance-threshold mayor."
        )

    return (
        np.concatenate(all_windows),
        np.concatenate(all_labels),
        pd.DataFrame(selection_rows),
    )


def main():
    args = parse_args()

    if args.distance_threshold < 0:
        raise ValueError("--distance-threshold debe ser mayor o igual que 0.")

    internal.base.MODELS_TO_RUN = args.models
    internal.base.EPOCHS = args.epochs
    internal.base.BATCH_SIZE = args.batch_size
    internal.base.check_dependencies()

    selected_files = [
        internal.DATA2_SCENE_FILES[scene_name] for scene_name in args.scenes
    ]
    data = internal.load_selected_scenes(selected_files)

    label_encoder = LabelEncoder()
    label_encoder.fit(data[internal.base.LABEL_COL])

    print("Escenas:", args.scenes)
    print("Clases globales:", [str(c) for c in label_encoder.classes_])
    print(f"Margen UWB: |d1-d2| <= {args.distance_threshold:g} cm")
    print("Configuraciones:", args.feature_sets)
    print("Modelos:", args.models)

    all_results = []
    global_true = {}
    global_pred = {}
    selection_summary = None

    for feature_set_name in args.feature_sets:
        feature_cols = FEATURE_SETS[feature_set_name]
        print("\n" + "#" * 70)
        print(f"Configuración: {feature_set_name} | columnas: {feature_cols}")
        print("#" * 70)

        X, y, current_selection = make_margin_windows(
            data=data,
            label_encoder=label_encoder,
            feature_cols=feature_cols,
            threshold=args.distance_threshold,
        )
        if selection_summary is None:
            selection_summary = current_selection

        results, y_true, y_pred = internal.run_internal_cv(
            X=X,
            y=y,
            models=args.models,
            folds=args.folds,
            label_encoder=label_encoder,
            ablation_name=feature_set_name,
            allow_missing_train_classes=True,
        )
        all_results.append(results)

        for model_name in args.models:
            key = (feature_set_name, model_name)
            global_true[key] = y_true[model_name]
            global_pred[key] = y_pred[model_name]

    results_df = pd.concat(all_results, ignore_index=True)
    os.makedirs(args.output_dir, exist_ok=True)

    selection_path = os.path.join(
        args.output_dir,
        "distance_margin_selection_summary.csv",
    )
    selection_summary.to_csv(selection_path, index=False)
    print(f"\nResumen de selección guardado en: {selection_path}")

    internal.save_outputs(
        results_df=results_df,
        global_true=global_true,
        global_pred=global_pred,
        label_encoder=label_encoder,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
