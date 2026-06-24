"""Internal 5-fold cross validation for the concatenated data2 dataset.

This script reuses the preprocessing/model definitions from
``model_cnn2lstm_xgb_svm_5scenes.py`` but evaluates data2 with an internal
StratifiedKFold split instead of cross-scene validation.

By default it ignores UWB/TSV availability and uses the CSV scenes as one
continuous dataset, because data2 currently stores only IMU/orientation values
in ``scene1.csv`` and ``scene2.csv``.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

import model_cnn2lstm_xgb_svm_5scenes as base


DATA2_SCENE_FILES = [
    "data2/scene1.csv",
    "data2/scene2.csv",
]

DEFAULT_MODELS = ["SVM", "XGBOOST"]
DEFAULT_OUTPUT_DIR = "results_data2_internal_cv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run internal StratifiedKFold validation on data2."
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
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSV outputs. Default: {DEFAULT_OUTPUT_DIR}.",
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


def load_data2_as_continuous_dataset(scene_files):
    """Load scene CSV files and concatenate them as a single internal dataset.

    Each scene is transformed with the same long-to-wide preprocessing used by
    the cross-scene script. Seconds are then offset so that the scenes are
    treated as consecutive blocks for window creation.
    """

    parts = []
    next_second = 0

    for path in scene_files:
        scene_name = os.path.splitext(os.path.basename(path))[0]
        df = base.load_scene_long_format(path).copy()
        df.insert(0, "scene", scene_name)
        df["local_second"] = df["second"]
        df["second"] = df["second"] + next_second
        next_second = int(df["second"].max()) + 1
        parts.append(df)

        label_counts = df[base.LABEL_COL].value_counts().sort_index().to_dict()
        print(
            f"{scene_name}: {len(df)} segundos cargados; "
            f"etiquetas por segundo: {label_counts}"
        )

    data = pd.concat(parts, axis=0, ignore_index=True)
    data = data.sort_values("second").reset_index(drop=True)
    return data


def make_internal_windows(data, label_encoder):
    X_raw = data[base.MODEL_COLS].values.astype(np.float32)
    y_raw = label_encoder.transform(data[base.LABEL_COL])

    # UWB/TSV is intentionally ignored here: data2 CSV does not contain real
    # distance_00_01/distance_00_02 samples and the request is to validate the
    # dataset internally, not to filter by the external TSV intervals.
    X_windows, y_windows, discarded, total, discard_pct = base.make_centered_windows(
        X_raw,
        y_raw,
        anchor_present=None,
        past=base.PAST_SECONDS,
        future=base.FUTURE_SECONDS,
    )

    print(
        f"Ventanas internas: {len(X_windows)} válidas de {total} candidatas; "
        f"descartadas por UWB: {discarded} ({discard_pct:.2f}%)"
    )
    return X_windows, y_windows


def run_internal_cv(X, y, models, folds, label_encoder):
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=base.RANDOM_SEED,
    )

    n_classes = len(label_encoder.classes_)
    fold_rows = []
    global_true = {model_name: [] for model_name in models}
    global_pred = {model_name: [] for model_name in models}

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        print("\n" + "=" * 70)
        print(f"Fold interno {fold_idx}/{folds}")
        print("=" * 70)

        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

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
                "model": model_name,
                "fold": fold_idx,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "n_train_samples": len(y_train),
                "n_test_samples": len(y_test),
            })

    return pd.DataFrame(fold_rows), global_true, global_pred


def save_outputs(results_df, global_true, global_pred, label_encoder, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    all_results_path = os.path.join(output_dir, "results_data2_internal_cv_all_models.csv")
    results_df.to_csv(all_results_path, index=False)
    print(f"\nResultados por fold guardados en: {all_results_path}")

    class_names = [str(c) for c in label_encoder.classes_]
    labels = np.arange(len(class_names))

    for model_name in sorted(global_true):
        safe_name = safe_model_name(model_name)
        model_results = results_df[results_df["model"] == model_name].copy()
        model_results_path = os.path.join(
            output_dir,
            f"results_data2_internal_cv_{safe_name}.csv",
        )
        model_results.to_csv(model_results_path, index=False)
        print(f"Resultados de {model_name} guardados en: {model_results_path}")

        y_true = np.array(global_true[model_name], dtype=np.int32)
        y_pred = np.array(global_pred[model_name], dtype=np.int32)

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
            f"classification_report_data2_internal_cv_{safe_name}.csv",
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
            f"confusion_matrix_data2_internal_cv_{safe_name}.csv",
        )
        cm_df.to_csv(cm_path)
        print(f"Matriz de confusión de {model_name} guardada en: {cm_path}")

        print("\n" + "=" * 70)
        print(f"Resumen global data2 internal CV - {model_name}")
        print("=" * 70)
        print(report_df)
        print(cm_df)


def main():
    args = parse_args()

    base.MODELS_TO_RUN = args.models
    base.EPOCHS = args.epochs
    base.BATCH_SIZE = args.batch_size
    base.check_dependencies()

    data = load_data2_as_continuous_dataset(DATA2_SCENE_FILES)

    label_encoder = LabelEncoder()
    label_encoder.fit(data[base.LABEL_COL])

    print("\nClases detectadas:", [str(c) for c in label_encoder.classes_])
    print("Modelos:", args.models)

    X, y = make_internal_windows(data, label_encoder)

    results_df, global_true, global_pred = run_internal_cv(
        X=X,
        y=y,
        models=args.models,
        folds=args.folds,
        label_encoder=label_encoder,
    )

    save_outputs(
        results_df=results_df,
        global_true=global_true,
        global_pred=global_pred,
        label_encoder=label_encoder,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
