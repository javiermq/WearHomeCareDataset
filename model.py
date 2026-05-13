import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout


# ============================================================
# iiConfiguración
# ============================================================

SCENE_FILES = [
    "data/scene1.csv",
 #   "data/scene2.csv",
    "data/scene3.csv",
    "data/scene4.csv",
]

RAW_COLUMNS = ["timestamp", "sensor", "label", "valor"]

FEATURE_COLS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "rot_w",
    "rot_accuracy",
]

ANCHOR_COLS = [
    "distance_00_01",
    "distance_00_02",
]

MODEL_COLS = FEATURE_COLS + ANCHOR_COLS

ANCHOR_PRESENT_COLS = [
    f"has_{c}" for c in ANCHOR_COLS
]

LABEL_COL = "label"

PAST_SECONDS = 5
FUTURE_SECONDS = 5

# 5 segundos pasados + instante actual + 5 segundos futuros = 11
WINDOW_SIZE = PAST_SECONDS + 1 + FUTURE_SECONDS

EPOCHS = 100
BATCH_SIZE = 2
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# Carga y transformación de formato largo a formato ancho
# ============================================================

def load_scene_long_format(path):
    """
    Lee un fichero tipo:

    timestamp,sensor,label,valor
    1778244788436,acc_x,0,-2.8051054
    1778244788436,acc_y,0,-2.4865067
    ...

    y lo transforma a una matriz por segundo:

    second, label, acc_x, ..., rot_accuracy, distance_00_01, distance_00_02

    Además añade columnas:
    has_distance_00_01, has_distance_00_02

    Estas columnas indican si realmente hubo muestra de esa distancia
    en ese segundo antes de aplicar ffill/bfill.
    """

    df = pd.read_csv(path)

    missing_cols = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas en {path}: {missing_cols}")

    df = df[RAW_COLUMNS].copy()

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")

    df = df.dropna(subset=["timestamp", "sensor", "label", "valor"])

    # Convertimos milisegundos a segundos relativos.
    # Esto agrupa todas las muestras que caen dentro del mismo segundo.
    df["second"] = ((df["timestamp"] - df["timestamp"].min()) // 1000).astype(int)

    # Etiqueta por segundo usando todos los sensores disponibles.
    # Si hubiera varias etiquetas dentro del mismo segundo, usamos la moda.
    y_per_second = (
        df.groupby("second")[LABEL_COL]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )

    # Nos quedamos solo con los sensores que queremos meter al modelo:
    # IMU/orientación + distancias UWB.
    df_sensors = df[df["sensor"].isin(MODEL_COLS)].copy()

    # Valores por segundo.
    # Si hay varias muestras del mismo sensor dentro del mismo segundo,
    # usamos la mediana.
    X_wide = (
        df_sensors.pivot_table(
            index="second",
            columns="sensor",
            values="valor",
            aggfunc="median"
        )
        .reset_index()
    )

    # Presencia REAL de las distancias por segundo, antes de rellenar.
    # Esto es importante porque luego haremos ffill/bfill para los valores,
    # pero no queremos confundir valores rellenados con muestras reales.
    anchor_df = df_sensors[df_sensors["sensor"].isin(ANCHOR_COLS)].copy()

    if len(anchor_df) > 0:
        anchor_presence = (
            anchor_df
            .assign(present=1)
            .pivot_table(
                index="second",
                columns="sensor",
                values="present",
                aggfunc="max",
                fill_value=0
            )
            .reset_index()
        )
    else:
        anchor_presence = pd.DataFrame({"second": sorted(df["second"].unique())})

    for col in ANCHOR_COLS:
        if col not in anchor_presence.columns:
            anchor_presence[col] = 0

    anchor_presence = anchor_presence[["second"] + ANCHOR_COLS]
    anchor_presence = anchor_presence.rename(
        columns={c: f"has_{c}" for c in ANCHOR_COLS}
    )

    # Merge de etiquetas + features + presencia real de anclas.
    data = pd.merge(y_per_second, X_wide, on="second", how="inner")
    data = pd.merge(data, anchor_presence, on="second", how="left")

    data = data.sort_values("second").reset_index(drop=True)

    # Aseguramos que todas las columnas existen.
    for col in MODEL_COLS:
        if col not in data.columns:
            data[col] = np.nan

    for col in ANCHOR_PRESENT_COLS:
        if col not in data.columns:
            data[col] = 0

    data[ANCHOR_PRESENT_COLS] = data[ANCHOR_PRESENT_COLS].fillna(0).astype(int)

    data = data[["second", LABEL_COL] + MODEL_COLS + ANCHOR_PRESENT_COLS]

    # Relleno de huecos en valores de sensores.
    # OJO: esto rellena los valores distance_00_01/distance_00_02,
    # pero NO modifica has_distance_00_01/has_distance_00_02.
    # Por eso podemos saber después si una ventana tenía muestras reales.
    data[MODEL_COLS] = data[MODEL_COLS].ffill().bfill()

    # Si aún queda algún NaN porque un canal no existe en todo el fichero,
    # lo sustituimos por 0.
    data[MODEL_COLS] = data[MODEL_COLS].fillna(0.0)

    return data


# ============================================================
# Ventanas deslizantes centradas con filtro UWB
# ============================================================

def make_centered_windows(X, y, anchor_present=None, past=5, future=5):
    """
    Crea ventanas centradas.

    Para cada instante i:

        [i-5, i-4, ..., i, ..., i+4, i+5]

    La etiqueta de la ventana es la etiqueta del instante central i.

    Si anchor_present no es None, descarta ventanas donde no haya
    al menos una muestra real de distance_00_01 y al menos una muestra
    real de distance_00_02 dentro de la ventana.
    """

    X_windows = []
    y_windows = []

    total_candidates = 0
    discarded = 0

    n = len(X)

    for i in range(past, n - future):
        total_candidates += 1

        start = i - past
        end = i + future + 1

        if anchor_present is not None:
            anchor_window = anchor_present[start:end]

            # anchor_present tiene forma:
            # columna 0 -> has_distance_00_01
            # columna 1 -> has_distance_00_02
            has_anchor_1 = np.any(anchor_window[:, 0] == 1)
            has_anchor_2 = np.any(anchor_window[:, 1] == 1)

            if not (has_anchor_1 and has_anchor_2):
                discarded += 1
                continue

        window = X[start:end]
        label = y[i]

        X_windows.append(window)
        y_windows.append(label)

    discard_pct = 100.0 * discarded / total_candidates if total_candidates > 0 else 0.0

    return (
        np.array(X_windows),
        np.array(y_windows),
        discarded,
        total_candidates,
        discard_pct
    )


# ============================================================
# Modelo LSTM sencillo
# ============================================================

def build_lstm_model(input_shape, n_classes):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(n_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# Cross-scene validation
# ============================================================

def cross_scene_validation(scene_files):
    scenes = []

    for path in scene_files:
        scene_name = os.path.splitext(os.path.basename(path))[0]
        df = load_scene_long_format(path)

        print(f"{scene_name}: {df.shape[0]} segundos cargados")

        # Info rápida de presencia de UWB por escena.
        for anchor_col, present_col in zip(ANCHOR_COLS, ANCHOR_PRESENT_COLS):
            n_seconds_with_anchor = int(df[present_col].sum())
            pct_seconds_with_anchor = 100.0 * n_seconds_with_anchor / len(df) if len(df) > 0 else 0.0

            print(
                f"  {anchor_col}: presente en "
                f"{n_seconds_with_anchor}/{len(df)} segundos "
                f"({pct_seconds_with_anchor:.2f}%)"
            )

        scenes.append((scene_name, df))

    all_labels = pd.concat([df[LABEL_COL] for _, df in scenes], axis=0)

    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    print("\nClases detectadas:", list(label_encoder.classes_))
    print("\nCanales usados por el modelo:")
    for col in MODEL_COLS:
        print(f"  - {col}")

    all_y_true = []
    all_y_pred = []

    fold_results = []

    for test_idx in range(len(scenes)):
        test_scene_name, test_df = scenes[test_idx]

        train_dfs = [
            df for idx, (_, df) in enumerate(scenes)
            if idx != test_idx
        ]

        train_df = pd.concat(train_dfs, axis=0).reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        print("\n" + "=" * 70)
        print(f"Fold: entrenar con todas menos {test_scene_name}")
        print(f"Test scene: {test_scene_name}")
        print("=" * 70)

        X_train_raw = train_df[MODEL_COLS].values.astype(np.float32)
        y_train_raw = label_encoder.transform(train_df[LABEL_COL])
        anchor_train_present = train_df[ANCHOR_PRESENT_COLS].values.astype(np.int32)

        X_test_raw = test_df[MODEL_COLS].values.astype(np.float32)
        y_test_raw = label_encoder.transform(test_df[LABEL_COL])
        anchor_test_present = test_df[ANCHOR_PRESENT_COLS].values.astype(np.int32)

        # ----------------------------------------------------
        # Normalización
        # ----------------------------------------------------
        # El scaler se ajusta SOLO con train.
        # Esto evita fuga de información desde la escena de test.
        # Se normaliza por canal/columna.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # ----------------------------------------------------
        # Ventanas
        # ----------------------------------------------------
        X_train, y_train, train_discarded, train_total, train_discard_pct = make_centered_windows(
            X_train_scaled,
            y_train_raw,
            anchor_present=anchor_train_present,
            past=PAST_SECONDS,
            future=FUTURE_SECONDS
        )

        X_test, y_test, test_discarded, test_total, test_discard_pct = make_centered_windows(
            X_test_scaled,
            y_test_raw,
            anchor_present=anchor_test_present,
            past=PAST_SECONDS,
            future=FUTURE_SECONDS
        )

        print(
            f"Descartadas train: {train_discarded}/{train_total} "
            f"({train_discard_pct:.2f}%) por falta de alguna distancia UWB en la ventana"
        )

        print(
            f"Descartadas test : {test_discarded}/{test_total} "
            f"({test_discard_pct:.2f}%) por falta de alguna distancia UWB en la ventana"
        )

        print("X_train:", X_train.shape)
        print("y_train:", y_train.shape)
        print("X_test :", X_test.shape)
        print("y_test :", y_test.shape)

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Saltando fold {test_scene_name}: no hay suficientes ventanas válidas.")
            continue

        n_classes = len(label_encoder.classes_)
        input_shape = X_train.shape[1], X_train.shape[2]

        model = build_lstm_model(
            input_shape=input_shape,
            n_classes=n_classes
        )

        # Sin EarlyStopping: entrenará siempre EPOCHS epochs.
        model.fit(
            X_train,
            y_train,
            validation_split=0.15,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

        print(f"Test loss: {test_loss:.4f}")
        print(f"Test acc : {test_acc:.4f}")

        y_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        fold_results.append({
            "test_scene": test_scene_name,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "n_test_samples": len(y_test),
            "train_windows_total": train_total,
            "train_windows_discarded": train_discarded,
            "train_discard_pct": train_discard_pct,
            "test_windows_total": test_total,
            "test_windows_discarded": test_discarded,
            "test_discard_pct": test_discard_pct
        })

    # ========================================================
    # Resultados globales
    # ========================================================

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    results_df = pd.DataFrame(fold_results)

    print("\n" + "=" * 70)
    print("Resultados por fold")
    print("=" * 70)
    print(results_df)

    results_df.to_csv("results_cross_scene.csv", index=False)
    print("\nResultados guardados en: results_cross_scene.csv")

    if len(all_y_true) == 0:
        print("\nNo hay predicciones globales porque todos los folds se han saltado.")
        return results_df, None, label_encoder

    print("\n" + "=" * 70)
    print("Classification report global")
    print("=" * 70)

    print(
        classification_report(
            all_y_true,
            all_y_pred,
            target_names=label_encoder.classes_
        )
    )

    cm = confusion_matrix(all_y_true, all_y_pred)

    print("\n" + "=" * 70)
    print("Matriz de confusión global")
    print("=" * 70)

    cm_df = pd.DataFrame(
        cm,
        index=[f"Real {c}" for c in label_encoder.classes_],
        columns=[f"Pred {c}" for c in label_encoder.classes_]
    )

    print(cm_df)

    cm_df.to_csv("confusion_matrix_cross_scene.csv")
    print("\nMatriz guardada en: confusion_matrix_cross_scene.csv")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )

    disp.plot(values_format="d")
    plt.title("Matriz de confusión global - Cross-scene validation")
    plt.tight_layout()
    plt.savefig("confusion_matrix_cross_scene.png", dpi=300)
    plt.close()

    print("Imagen guardada en: confusion_matrix_cross_scene.png")

    return results_df, cm, label_encoder


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    results_df, cm, label_encoder = cross_scene_validation(SCENE_FILES)