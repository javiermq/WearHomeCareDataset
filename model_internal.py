"""Internal blocked 5-fold cross validation for the data2 dataset.

This script reuses the preprocessing/model definitions from
``model_cnn2lstm_xgb_svm_5scenes.py`` but evaluates data2 with an internal
cross-validation split made of contiguous temporal blocks.

It aligns the two UWB TSV files with the CSV scenes by timestamp and compares
UWB+magnetometer, magnetometer-only and UWB-only feature sets. Use ``--scenes``
to select one or both scenes.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

import model_cnn2lstm_xgb_svm_5scenes as base


DATA2_SCENE_FILES = {
    "scene1": "data2/scene1.csv",
    "scene2": "data2/scene2.csv",
    "scene_interleaved": "data2/scene_interleaved.csv",
}

DEFAULT_SCENES = ["scene1", "scene2"]

DATA2_UWB_FILES = {
    "distance_00_01": "data2/20627.1.tsv",
    "distance_00_02": "data2/20627.2.tsv",
}

ABLATION_FEATURES = {
    "UWB_MAG": base.ANCHOR_COLS + ["mag_x", "mag_y", "mag_z"],
    "MAG": ["mag_x", "mag_y", "mag_z"],
    "UWB": base.ANCHOR_COLS,
}

DEFAULT_MODELS = ["SVM", "XGBOOST"]
DEFAULT_ABLATIONS = list(ABLATION_FEATURES)
DEFAULT_OUTPUT_DIR = "results_model_internal_data2"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run internal blocked cross-validation on data2."
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of internal CV folds. Default: 5.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "Models to run. Supported by the base module: SVM, XGBOOST, "
            "SIMPLE_LSTM, STACKED_LSTM, CNN_2LSTM. Default: SVM XGBOOST."
        ),
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=DEFAULT_SCENES,
        help=(
            "Scene aliases or CSV paths, with or without .csv. Examples: "
            "scene1, data2/scene1, data3/sceneAB."
        ),
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        type=str.upper,
        choices=tuple(ABLATION_FEATURES),
        default=DEFAULT_ABLATIONS,
        help="Feature sets to test. Default: UWB_MAG MAG UWB.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            f"Directory for CSV outputs. Default: {DEFAULT_OUTPUT_DIR} when "
            "using both scenes, or a scene-specific directory when using one."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=base.EPOCHS,
        help="Epochs for neural models. Default: value from base model script.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=base.BATCH_SIZE,
        help="Batch size for neural models. Default: value from base model script.",
    )
    return parser.parse_args()


def safe_model_name(model_name):
    return model_name.lower().replace(" ", "_").replace("+", "plus")


def _normalized_scene_key(value):
    key = os.path.splitext(os.path.basename(value))[0].lower()
    key = re.sub(r"^(scene|sensor)[_-]*", "", key)
    return re.sub(r"[^a-z0-9]+", "", key)


def resolve_scene_path(scene_spec):
    """Resolve an alias or extensionless path to an existing scene CSV."""
    if scene_spec in DATA2_SCENE_FILES:
        return DATA2_SCENE_FILES[scene_spec]

    candidates = [scene_spec]
    if not scene_spec.lower().endswith(".csv"):
        candidates.append(f"{scene_spec}.csv")
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.normpath(candidate)

    directory = os.path.dirname(scene_spec) or "."
    if os.path.isdir(directory):
        requested_key = _normalized_scene_key(scene_spec)
        matching = [
            os.path.join(directory, filename)
            for filename in os.listdir(directory)
            if filename.lower().endswith(".csv")
            and _normalized_scene_key(filename) == requested_key
        ]
        if len(matching) == 1:
            return os.path.normpath(matching[0])

    raise FileNotFoundError(
        f"No se pudo resolver la escena '{scene_spec}' a un fichero CSV."
    )


def discover_uwb_files(scene_csv_path):
    """Find the two anchor TSV files located beside a scene CSV."""
    directory = os.path.dirname(scene_csv_path) or "."
    tsv_files = sorted(
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.lower().endswith(".tsv")
    )
    if len(tsv_files) != 2:
        raise ValueError(
            f"{scene_csv_path}: se esperaban exactamente 2 TSV UWB en "
            f"{directory}, pero se encontraron {len(tsv_files)}."
        )
    return dict(zip(base.ANCHOR_COLS, tsv_files))


def load_uwb_by_second(scene_csv_path, uwb_files=None):
    """Align both UWB anchors to one scene using local Europe/Madrid time."""
    raw_timestamps = pd.read_csv(
        scene_csv_path,
        usecols=["timestamp"],
    )["timestamp"]
    scene_start_utc = pd.to_datetime(
        pd.to_numeric(raw_timestamps, errors="coerce").min(),
        unit="ms",
        utc=True,
    )
    scene_start_local = scene_start_utc.tz_convert("Europe/Madrid").tz_localize(None)

    aligned = None
    if uwb_files is None:
        uwb_files = discover_uwb_files(scene_csv_path)

    for distance_col, path in uwb_files.items():
        uwb = pd.read_csv(path, sep="\t", usecols=["timestamp", "distance_cm"])
        uwb["timestamp"] = pd.to_datetime(uwb["timestamp"], errors="coerce")
        uwb["distance_cm"] = pd.to_numeric(uwb["distance_cm"], errors="coerce")
        uwb = uwb.dropna(subset=["timestamp", "distance_cm"])
        uwb["second"] = (
            (uwb["timestamp"] - scene_start_local).dt.total_seconds() // 1
        ).astype(int)
        uwb = (
            uwb.groupby("second", as_index=False)["distance_cm"]
            .median()
            .rename(columns={"distance_cm": distance_col})
        )
        uwb[f"has_{distance_col}"] = 1

        if aligned is None:
            aligned = uwb
        else:
            aligned = pd.merge(aligned, uwb, on="second", how="outer")

    return aligned


def load_selected_scenes(scene_files, require_uwb=True):
    """Load selected scene CSV files and concatenate them in the given order.

    Each scene is transformed with the same long-to-wide preprocessing used by
    the cross-scene script. Seconds are offset for ordering, but window creation
    remains separate per scene.
    """

    parts = []
    next_second = 0

    for path in scene_files:
        scene_name = os.path.splitext(os.path.basename(path))[0]
        df = base.load_scene_long_format(path).copy()
        if require_uwb:
            has_embedded_uwb = all(
                int(df[present_col].sum()) > 0
                for present_col in base.ANCHOR_PRESENT_COLS
            )
            if not has_embedded_uwb:
                uwb = load_uwb_by_second(path)
                df = df.drop(
                    columns=base.ANCHOR_COLS + base.ANCHOR_PRESENT_COLS,
                    errors="ignore",
                )
                df = pd.merge(df, uwb, on="second", how="left")
                df[base.ANCHOR_PRESENT_COLS] = (
                    df[base.ANCHOR_PRESENT_COLS].fillna(0).astype(int)
                )
                df[base.ANCHOR_COLS] = df[base.ANCHOR_COLS].ffill().bfill()

            missing_uwb = df[base.ANCHOR_COLS].isna().any(axis=1)
            if missing_uwb.any():
                raise ValueError(
                    f"{scene_name}: no hay UWB alineado con esta grabación. "
                    "Usa --ablations MAG o añade los TSV correspondientes."
                )

        df.insert(0, "scene", scene_name)
        df["local_second"] = df["second"]
        df["second"] = df["second"] + next_second
        next_second = int(df["second"].max()) + 1
        parts.append(df)

        label_counts = df[base.LABEL_COL].value_counts().sort_index().to_dict()
        uwb_counts = {
            col: int(df[f"has_{col}"].sum()) for col in base.ANCHOR_COLS
        }
        print(
            f"{scene_name}: {len(df)} segundos cargados; "
            f"etiquetas por segundo: {label_counts}; "
            f"segundos con UWB real: {uwb_counts}"
        )

    data = pd.concat(parts, axis=0, ignore_index=True)
    data = data.sort_values("second").reset_index(drop=True)
    return data


def make_internal_windows(data, label_encoder, feature_cols):
    # Build windows per scene so none crosses a scene boundary.
    scene_windows = []
    scene_labels = []

    for scene_name, scene_data in data.groupby("scene", sort=False):
        X_raw = scene_data[feature_cols].values.astype(np.float32)
        y_raw = label_encoder.transform(scene_data[base.LABEL_COL])
        X_windows, y_windows, discarded, total, discard_pct = (
            base.make_centered_windows(
                X_raw,
                y_raw,
                anchor_present=None,
                past=base.PAST_SECONDS,
                future=base.FUTURE_SECONDS,
            )
        )
        scene_windows.append(X_windows)
        scene_labels.append(y_windows)
        print(
            f"{scene_name}: {len(X_windows)} ventanas válidas de {total}"
        )

    return np.concatenate(scene_windows), np.concatenate(scene_labels)


def run_internal_cv(
    X,
    y,
    models,
    folds,
    label_encoder,
    ablation_name,
    allow_missing_train_classes=False,
):
    if folds < 2:
        raise ValueError("--folds debe ser al menos 2.")
    if folds > len(y):
        raise ValueError(
            f"--folds ({folds}) no puede superar el número de ventanas ({len(y)})."
        )

    splitter = KFold(n_splits=folds, shuffle=False)

    n_classes = len(label_encoder.classes_)
    all_classes = set(range(n_classes))
    fold_rows = []
    global_true = {model_name: [] for model_name in models}
    global_pred = {model_name: [] for model_name in models}

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        print("\n" + "=" * 70)
        print(f"Ablación {ablation_name} | Fold interno {fold_idx}/{folds}")
        print("=" * 70)

        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        train_classes = set(np.unique(y_train).tolist())
        test_classes = set(np.unique(y_test).tolist())
        train_names = [str(label_encoder.classes_[i]) for i in sorted(train_classes)]
        test_names = [str(label_encoder.classes_[i]) for i in sorted(test_classes)]

        print(
            f"Bloque test contiguo: ventanas {test_idx[0]}..{test_idx[-1]} | "
            f"clases train={train_names} | clases test={test_names}"
        )
        missing_test = all_classes - test_classes
        if missing_test:
            missing_names = [
                str(label_encoder.classes_[i]) for i in sorted(missing_test)
            ]
            print(
                "AVISO: este bloque temporal no contiene las clases "
                f"{missing_names}."
            )
        missing_train = all_classes - train_classes
        if missing_train:
            missing_names = [
                str(label_encoder.classes_[i]) for i in sorted(missing_train)
            ]
            message = (
                "El conjunto de entrenamiento del fold no contiene las clases "
                f"{missing_names}."
            )
            if allow_missing_train_classes:
                print(f"AVISO: {message}")
            else:
                raise ValueError(
                    f"{message} No se puede entrenar de forma fiable."
                )

        scaler = StandardScaler()
        n_train, window_size, n_features = X_train_raw.shape
        n_test = X_test_raw.shape[0]

        X_train_flat_time = X_train_raw.reshape(-1, n_features)
        X_test_flat_time = X_test_raw.reshape(-1, n_features)

        X_train_scaled = scaler.fit_transform(X_train_flat_time).reshape(
            n_train,
            window_size,
            n_features,
        )
        X_test_scaled = scaler.transform(X_test_flat_time).reshape(
            n_test,
            window_size,
            n_features,
        )

        for model_name in models:
            print("\n" + "-" * 70)
            print(f"Modelo: {model_name} | Fold interno: {fold_idx}/{folds}")
            print("-" * 70)

            result = base.train_predict_model(
                model_name=model_name,
                X_train=X_train_scaled,
                y_train=y_train,
                X_test=X_test_scaled,
                y_test=y_test,
                n_classes=n_classes,
            )

            y_pred = result["y_pred"]
            test_loss = result["test_loss"]
            test_acc = result["test_acc"]

            print(f"Test loss: {test_loss:.4f}")
            print(f"Test acc : {test_acc:.4f}")

            global_true[model_name].extend(y_test.tolist())
            global_pred[model_name].extend(y_pred.tolist())

            fold_rows.append({
                "ablation": ablation_name,
                "model": model_name,
                "fold": fold_idx,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "n_train_samples": len(y_train),
                "n_test_samples": len(y_test),
                "test_start_index": int(test_idx[0]),
                "test_end_index": int(test_idx[-1]),
                "test_classes": ",".join(test_names),
            })

    return pd.DataFrame(fold_rows), global_true, global_pred


def save_outputs(results_df, global_true, global_pred, label_encoder, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    all_results_path = os.path.join(
        output_dir,
        "results_model_internal_data2_all_models.csv",
    )
    results_df.to_csv(all_results_path, index=False)
    print(f"\nResultados por fold guardados en: {all_results_path}")

    summary_df = (
        results_df.groupby(["ablation", "model"], as_index=False)
        .agg(
            mean_test_acc=("test_acc", "mean"),
            std_test_acc=("test_acc", "std"),
            mean_test_loss=("test_loss", "mean"),
            std_test_loss=("test_loss", "std"),
            folds=("fold", "count"),
        )
        .sort_values(["mean_test_acc", "ablation", "model"], ascending=[False, True, True])
    )
    summary_path = os.path.join(
        output_dir,
        "summary_model_internal_data2_ablation.csv",
    )
    summary_df.to_csv(summary_path, index=False)
    print(f"Resumen comparativo guardado en: {summary_path}")

    class_names = [str(c) for c in label_encoder.classes_]
    labels = np.arange(len(class_names))

    for ablation_name, model_name in sorted(global_true):
        safe_name = safe_model_name(model_name)
        safe_ablation = safe_model_name(ablation_name)
        model_results = results_df[
            (results_df["ablation"] == ablation_name)
            & (results_df["model"] == model_name)
        ].copy()
        model_results_path = os.path.join(
            output_dir,
            f"results_model_internal_data2_{safe_ablation}_{safe_name}.csv",
        )
        model_results.to_csv(model_results_path, index=False)
        print(f"Resultados de {model_name} guardados en: {model_results_path}")

        result_key = (ablation_name, model_name)
        y_true = np.array(global_true[result_key], dtype=np.int32)
        y_pred = np.array(global_pred[result_key], dtype=np.int32)

        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        )
        report_df = pd.DataFrame(report).transpose()
        report_path = os.path.join(
            output_dir,
            f"classification_report_model_internal_data2_"
            f"{safe_ablation}_{safe_name}.csv",
        )
        report_df.to_csv(report_path)
        print(f"Classification report de {model_name} guardado en: {report_path}")

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(
            cm,
            index=[f"Real {c}" for c in class_names],
            columns=[f"Pred {c}" for c in class_names],
        )
        cm_path = os.path.join(
            output_dir,
            f"confusion_matrix_model_internal_data2_"
            f"{safe_ablation}_{safe_name}.csv",
        )
        cm_df.to_csv(cm_path)
        print(f"Matriz de confusión de {model_name} guardada en: {cm_path}")

        print("\n" + "=" * 70)
        print(
            f"Resumen global data2 internal CV - "
            f"{ablation_name} - {model_name}"
        )
        print("=" * 70)
        print(report_df)
        print(cm_df)


def main():
    args = parse_args()

    base.MODELS_TO_RUN = args.models
    base.EPOCHS = args.epochs
    base.BATCH_SIZE = args.batch_size
    base.check_dependencies()

    selected_scene_files = [DATA2_SCENE_FILES[name] for name in args.scenes]
    print("Escenas seleccionadas:", args.scenes)
    require_uwb = any(
        any(col in base.ANCHOR_COLS for col in ABLATION_FEATURES[name])
        for name in args.ablations
    )
    data = load_selected_scenes(selected_scene_files, require_uwb=require_uwb)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
        if args.scenes != DEFAULT_SCENES:
            output_dir += "_" + "_".join(args.scenes)

    label_encoder = LabelEncoder()
    label_encoder.fit(data[base.LABEL_COL])

    print("\nClases detectadas:", [str(c) for c in label_encoder.classes_])
    print("Número de clases detectadas:", len(label_encoder.classes_))
    print("Modelos:", args.models)
    print("Ablaciones:", args.ablations)

    all_results = []
    global_true = {}
    global_pred = {}

    for ablation_name in args.ablations:
        feature_cols = ABLATION_FEATURES[ablation_name]
        print("\n" + "#" * 70)
        print(f"Ablación: {ablation_name} | columnas: {feature_cols}")
        print("#" * 70)

        X, y = make_internal_windows(data, label_encoder, feature_cols)
        ablation_results, ablation_true, ablation_pred = run_internal_cv(
            X=X,
            y=y,
            models=args.models,
            folds=args.folds,
            label_encoder=label_encoder,
            ablation_name=ablation_name,
        )
        all_results.append(ablation_results)
        for model_name in args.models:
            key = (ablation_name, model_name)
            global_true[key] = ablation_true[model_name]
            global_pred[key] = ablation_pred[model_name]

    results_df = pd.concat(all_results, ignore_index=True)

    save_outputs(
        results_df=results_df,
        global_true=global_true,
        global_pred=global_pred,
        label_encoder=label_encoder,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
