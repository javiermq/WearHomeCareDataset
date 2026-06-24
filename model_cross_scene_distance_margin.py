"""Cross-scene classification where both UWB distances are similar.

For every fold, one complete scene is used as test and all remaining scenes
are used as train. Only windows whose central instant satisfies

    abs(distance_00_01 - distance_00_02) <= --distance-threshold

are retained. Optionally, ``--require-real-uwb`` can also require at least one
real reading from each UWB anchor inside the window. The experiment compares
UWB-only against UWB plus magnetometer (compass). Distances and threshold are
expressed in centimetres.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

import model_cnn2lstm_xgb_svm_5scenes as base


SCENE_FILES = {
    f"scene{i}": f"data/scene{i}.csv"
    for i in range(1, 6)
}

DEFAULT_SCENES = [
    os.path.splitext(os.path.basename(path))[0]
    for path in base.SCENE_FILES
]

FEATURE_SETS = {
    "UWB": base.ANCHOR_COLS,
    "UWB_BRUJULA": base.ANCHOR_COLS + ["mag_x", "mag_y", "mag_z"],
}

DEFAULT_MODELS = ["SVM", "XGBOOST"]
DEFAULT_OUTPUT_DIR = "results_cross_scene_distance_margin"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Leave-one-scene-out comparison of UWB and UWB+compass where "
            "both UWB distances are similar."
        )
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        required=True,
        help="Maximum |distance_00_01-distance_00_02| in centimetres.",
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
        "--scenes",
        nargs="+",
        choices=tuple(SCENE_FILES),
        default=DEFAULT_SCENES,
        help=(
            "Scenes participating in leave-one-scene-out. "
            "Default: scene1 scene2 scene3 scene4, matching the base script."
        ),
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
        "--require-real-uwb",
        action="store_true",
        help=(
            "Also require at least one real reading from each UWB anchor "
            "inside every selected window."
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


def load_scenes(scene_names):
    scenes = []
    for scene_name in scene_names:
        data = base.load_scene_long_format(SCENE_FILES[scene_name])
        scenes.append((scene_name, data))
        print(
            f"{scene_name}: {len(data)} segundos; clases: "
            f"{data[base.LABEL_COL].value_counts().sort_index().to_dict()}"
        )
    return scenes


def make_margin_windows(
    scene_data,
    label_encoder,
    feature_cols,
    threshold,
    require_real_uwb=False,
):
    """Create windows inside one scene and filter ambiguous UWB centres."""
    scene_data = scene_data.reset_index(drop=True)
    X_raw = scene_data[feature_cols].to_numpy(dtype=np.float32)
    y_raw = label_encoder.transform(scene_data[base.LABEL_COL])
    distances = scene_data[base.ANCHOR_COLS].to_numpy(dtype=np.float32)
    anchor_present = scene_data[base.ANCHOR_PRESENT_COLS].to_numpy(dtype=np.int32)

    X_windows = []
    y_windows = []
    differences = []
    past = base.PAST_SECONDS
    future = base.FUTURE_SECONDS
    candidates = max(0, len(scene_data) - past - future)
    rejected_margin = 0
    rejected_uwb_presence = 0

    for center in range(past, len(scene_data) - future):
        start = center - past
        end = center + future + 1
        difference = float(abs(distances[center, 0] - distances[center, 1]))

        if difference > threshold:
            rejected_margin += 1
            continue

        presence_window = anchor_present[start:end]
        has_both_real_anchors = (
            np.any(presence_window[:, 0] == 1)
            and np.any(presence_window[:, 1] == 1)
        )
        if require_real_uwb and not has_both_real_anchors:
            rejected_uwb_presence += 1
            continue

        X_windows.append(X_raw[start:end])
        y_windows.append(y_raw[center])
        differences.append(difference)

    if X_windows:
        X_windows = np.asarray(X_windows, dtype=np.float32)
        y_windows = np.asarray(y_windows, dtype=np.int32)
        decoded = label_encoder.inverse_transform(y_windows)
        class_counts = pd.Series(decoded).value_counts().sort_index().to_dict()
    else:
        X_windows = np.empty(
            (0, base.WINDOW_SIZE, len(feature_cols)),
            dtype=np.float32,
        )
        y_windows = np.empty(0, dtype=np.int32)
        class_counts = {}

    stats = {
        "candidate_windows": candidates,
        "selected_windows": len(y_windows),
        "selected_pct": 100.0 * len(y_windows) / max(1, candidates),
        "rejected_by_margin": rejected_margin,
        "rejected_by_uwb_presence": rejected_uwb_presence,
        "mean_distance_difference_cm": (
            float(np.mean(differences)) if differences else np.nan
        ),
        "max_distance_difference_cm": (
            float(np.max(differences)) if differences else np.nan
        ),
        "class_counts": str(class_counts),
    }
    return X_windows, y_windows, stats


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

    for feature_set_name in args.feature_sets:
        feature_cols = FEATURE_SETS[feature_set_name]
        windows_by_scene = {}

        print("\n" + "#" * 70)
        print(f"Configuración: {feature_set_name} | columnas: {feature_cols}")
        print("#" * 70)

        for scene_name, scene_data in scenes:
            X, y, stats = make_margin_windows(
                scene_data,
                label_encoder,
                feature_cols,
                args.distance_threshold,
                args.require_real_uwb,
            )
            windows_by_scene[scene_name] = (X, y)
            selection_rows.append({
                "feature_set": feature_set_name,
                "scene": scene_name,
                "distance_threshold_cm": args.distance_threshold,
                **stats,
            })
            print(
                f"{scene_name}: {len(y)} ventanas seleccionadas; "
                f"clases: {stats['class_counts']}"
            )

        for model_name in args.models:
            global_true[(feature_set_name, model_name)] = []
            global_pred[(feature_set_name, model_name)] = []

        for test_scene_name, _ in scenes:
            X_test, y_test = windows_by_scene[test_scene_name]
            train_parts = [
                windows_by_scene[name]
                for name, _ in scenes
                if name != test_scene_name
            ]
            nonempty_train = [(X, y) for X, y in train_parts if len(y) > 0]

            if len(y_test) == 0 or not nonempty_train:
                print(
                    f"Saltando test={test_scene_name}: no hay ventanas "
                    "suficientes con este margen."
                )
                continue

            X_train = np.concatenate([X for X, _ in nonempty_train])
            y_train = np.concatenate([y for _, y in nonempty_train])
            X_train, X_test_scaled = scale_windows(X_train, X_test)

            train_names = label_encoder.inverse_transform(np.unique(y_train)).tolist()
            test_names = label_encoder.inverse_transform(np.unique(y_test)).tolist()
            print("\n" + "=" * 70)
            print(
                f"{feature_set_name} | train=todas menos {test_scene_name} | "
                f"test={test_scene_name}"
            )
            print(
                f"Ventanas train={len(y_train)}, test={len(y_test)} | "
                f"clases train={train_names}, test={test_names}"
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
                key = (feature_set_name, model_name)
                global_true[key].extend(y_test.tolist())
                global_pred[key].extend(y_pred.tolist())
                fold_rows.append({
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "test_scene": test_scene_name,
                    "distance_threshold_cm": args.distance_threshold,
                    "test_loss": result["test_loss"],
                    "test_acc": result["test_acc"],
                    "n_train_samples": len(y_train),
                    "n_test_samples": len(y_test),
                    "train_classes": ",".join(map(str, train_names)),
                    "test_classes": ",".join(map(str, test_names)),
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
    results_df,
    selection_df,
    global_true,
    global_pred,
    label_encoder,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(
        os.path.join(output_dir, "results_cross_scene_distance_margin.csv"),
        index=False,
    )
    selection_df.to_csv(
        os.path.join(output_dir, "distance_margin_selection_by_scene.csv"),
        index=False,
    )

    summary = (
        results_df.groupby(["feature_set", "model"], as_index=False)
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
        os.path.join(output_dir, "summary_cross_scene_distance_margin.csv"),
        index=False,
    )

    class_names = [str(c) for c in label_encoder.classes_]
    labels = np.arange(len(class_names))
    for (feature_set_name, model_name), y_true_values in sorted(global_true.items()):
        if not y_true_values:
            continue
        safe_feature = feature_set_name.lower()
        safe_model = model_name.lower()
        y_true = np.asarray(y_true_values, dtype=np.int32)
        y_pred = np.asarray(
            global_pred[(feature_set_name, model_name)],
            dtype=np.int32,
        )
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
                output_dir,
                f"classification_report_{safe_feature}_{safe_model}.csv",
            )
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        pd.DataFrame(
            cm,
            index=[f"Real {name}" for name in class_names],
            columns=[f"Pred {name}" for name in class_names],
        ).to_csv(
            os.path.join(
                output_dir,
                f"confusion_matrix_{safe_feature}_{safe_model}.csv",
            )
        )

    print(f"\nResultados guardados en: {output_dir}")
    print(summary)


def main():
    args = parse_args()
    if args.distance_threshold < 0:
        raise ValueError("--distance-threshold debe ser mayor o igual que 0.")
    if len(args.scenes) < 2:
        raise ValueError("Cross-scene requiere al menos dos escenas.")

    base.MODELS_TO_RUN = args.models
    base.EPOCHS = args.epochs
    base.BATCH_SIZE = args.batch_size
    base.check_dependencies()

    scenes = load_scenes(args.scenes)
    all_labels = pd.concat(
        [scene_data[base.LABEL_COL] for _, scene_data in scenes],
        ignore_index=True,
    )
    label_encoder = LabelEncoder().fit(all_labels)

    print("Escenas:", args.scenes)
    print("Clases:", [str(c) for c in label_encoder.classes_])
    print(f"Margen: |d1-d2| <= {args.distance_threshold:g} cm")
    print("Configuraciones:", args.feature_sets)
    print("Modelos:", args.models)
    print("Exigir UWB real en cada ventana:", args.require_real_uwb)

    results, selection, global_true, global_pred = run_cross_scene(
        scenes,
        label_encoder,
        args,
    )
    if results.empty:
        raise ValueError(
            "No se pudo evaluar ningún fold. Prueba un margen mayor "
            "o selecciona más escenas."
        )
    save_outputs(
        results,
        selection,
        global_true,
        global_pred,
        label_encoder,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
