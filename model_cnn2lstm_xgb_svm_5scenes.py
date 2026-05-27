import os
import warnings
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    log_loss,
)
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Input,
        Conv1D,
        LayerNormalization,
        LSTM,
        Dense,
        Dropout,
    )
    from tensorflow.keras.optimizers import Adam
except ImportError:
    tf = None
    Sequential = None
    Input = None
    Conv1D = None
    LayerNormalization = None
    LSTM = None
    Dense = None
    Dropout = None
    Adam = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


# ============================================================
# Configuración
# ============================================================

SCENE_FILES = [
    "data/scene1.csv",
    "data/scene2.csv",
    "data/scene3.csv",
    "data/scene4.csv",
#    "data/scene5.csv",
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

# Modelos a ejecutar.
# Si quieres probar solo uno, deja únicamente su nombre en esta lista.
MODELS_TO_RUN = ["SIMPLE_LSTM", "STACKED_LSTM", "CNN_2LSTM", "XGBOOST", "SVM"]

# Si está en True, el script se para si falta alguna librería necesaria
# para los modelos seleccionados. Así no se generan métricas incompletas.
REQUIRE_ALL_SELECTED_MODELS = True

EPOCHS = 40
BATCH_SIZE = 2
RANDOM_SEED = 42

OUTPUT_DIR = "results_cross_scene_models"

np.random.seed(RANDOM_SEED)
if tf is not None:
    tf.random.set_seed(RANDOM_SEED)


# ============================================================
# Utilidades
# ============================================================

def safe_model_name(model_name):
    return model_name.lower().replace(" ", "_").replace("+", "plus")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def check_dependencies():
    missing = []

    if any(m in MODELS_TO_RUN for m in ["SIMPLE_LSTM", "CNN_2LSTM", "STACKED_LSTM", "LSTM"]) and tf is None:
        missing.append("tensorflow")

    if "XGBOOST" in MODELS_TO_RUN and XGBClassifier is None:
        missing.append("xgboost")

    if missing and REQUIRE_ALL_SELECTED_MODELS:
        raise ImportError(
            "Faltan librerías para ejecutar todos los modelos seleccionados: "
            + ", ".join(missing)
            + "\nInstala, por ejemplo:\n"
            + "pip install tensorflow xgboost scikit-learn pandas matplotlib"
        )

    if missing:
        warnings.warn(
            "Se saltarán modelos porque faltan librerías: " + ", ".join(missing)
        )


def flatten_windows(X_windows):
    """
    Convierte ventanas 3D:
        n_samples, window_size, n_features
    a una matriz 2D:
        n_samples, window_size * n_features

    Esto es lo que necesitan XGBoost y SVM.
    """
    return X_windows.reshape(X_windows.shape[0], -1)


def align_proba_to_global_classes(y_prob_local, local_classes, n_global_classes):
    """
    Algunos modelos solo devuelven probabilidades para las clases vistas en train.
    Esta función recoloca esas probabilidades en el espacio global de clases.

    Ejemplo:
        clases globales codificadas: [0, 1, 2]
        clases vistas en train:      [0, 2]

    La salida tendrá 3 columnas. La clase no vista se queda con probabilidad 0.
    """
    y_prob_global = np.zeros((y_prob_local.shape[0], n_global_classes), dtype=np.float64)

    for local_idx, global_class in enumerate(local_classes):
        y_prob_global[:, int(global_class)] = y_prob_local[:, local_idx]

    return y_prob_global


def compute_loss_from_proba(y_true, y_prob, n_classes):
    """
    Calcula log-loss de forma segura usando todas las clases globales.
    """
    return log_loss(
        y_true,
        y_prob,
        labels=np.arange(n_classes),
    )


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

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No existe el fichero {path}. Revisa SCENE_FILES o la carpeta data/."
        )

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
            aggfunc="median",
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
                fill_value=0,
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
        discard_pct,
    )

def build_simple_lstm_model(input_shape, n_classes):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
    
# ============================================================
# Modelos
# ============================================================

def build_cnn_2lstm_model(input_shape, n_classes):
    """
    Arquitectura temporal más sólida que una LSTM simple.

    Entrada:
        window_size x n_features

    Bloques:
        1) Conv1D para extraer patrones locales en ventanas cortas.
        2) Dos LSTM apiladas para modelar la dinámica temporal.
        3) Capa densa final para clasificación multiclase.

    Con WINDOW_SIZE=11, la convolución mira cambios locales entre segundos
    consecutivos y las LSTM integran la información pasada, central y futura.
    """
    model = Sequential([
        Input(shape=input_shape),

        Conv1D(32, kernel_size=3, padding="same", activation="relu"),
        LayerNormalization(),
        Dropout(0.20),

        Conv1D(64, kernel_size=3, padding="same", activation="relu"),
        LayerNormalization(),
        Dropout(0.20),

        LSTM(128, return_sequences=True),
        Dropout(0.30),

        LSTM(64, return_sequences=False),
        Dropout(0.30),

        Dense(64, activation="relu"),
        Dropout(0.20),
        Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_stacked_lstm_model(input_shape, n_classes):
    """
    Alternativa sin convolución: dos LSTM apiladas.
    Para usarla, cambia MODELS_TO_RUN a ["STACKED_LSTM", "XGBOOST", "SVM"].
    """
    model = Sequential([
        Input(shape=input_shape),

        LSTM(128, return_sequences=True),
        Dropout(0.30),

        LSTM(64, return_sequences=False),
        Dropout(0.30),

        Dense(64, activation="relu"),
        Dropout(0.20),
        Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_xgboost_model(n_local_classes):
    """
    XGBoost base para ventanas aplanadas.
    Usa multi:softprob si hay más de dos clases.
    """
    params = {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    if n_local_classes == 2:
        params.update({
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        })
    else:
        params.update({
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "num_class": n_local_classes,
        })

    return XGBClassifier(**params)


def build_svm_model():
    """
    SVM no lineal base.
    RBF suele ser la opción estándar cuando no asumimos frontera lineal.
    probability=True permite calcular log-loss y conservar métricas comparables.
    """
    return SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=RANDOM_SEED,
    )


def train_predict_deep_model(model_name, X_train, y_train, X_test, y_test, n_classes):
    if tf is None:
        raise ImportError("TensorFlow no está instalado y el modelo neuronal está seleccionado.")

    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)

    input_shape = X_train.shape[1], X_train.shape[2]

    if model_name == "SIMPLE_LSTM":
        model = build_simple_lstm_model(input_shape=input_shape, n_classes=n_classes)
    elif model_name == "CNN_2LSTM":
        model = build_cnn_2lstm_model(input_shape=input_shape, n_classes=n_classes)
    elif model_name == "STACKED_LSTM":
        model = build_stacked_lstm_model(input_shape=input_shape, n_classes=n_classes)
    elif model_name == "LSTM":
        # Compatibilidad: dejamos LSTM como la simple para evitar confusión
        model = build_simple_lstm_model(input_shape=input_shape, n_classes=n_classes)
    else:
        raise ValueError(f"Modelo neuronal no reconocido: {model_name}")

    model.summary()

    # Sin EarlyStopping: entrenará siempre EPOCHS epochs.
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    return {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def train_predict_xgboost(X_train, y_train, X_test, y_test, n_classes):
    if XGBClassifier is None:
        raise ImportError("xgboost no está instalado y XGBOOST está seleccionado.")

    X_train_flat = flatten_windows(X_train)
    X_test_flat = flatten_windows(X_test)

    # XGBoost necesita etiquetas locales consecutivas si en un fold falta alguna clase.
    local_classes = np.unique(y_train)

    if len(local_classes) == 1:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train_flat, y_train)
        y_pred = model.predict(X_test_flat)
        y_prob_local = model.predict_proba(X_test_flat)
        y_prob = align_proba_to_global_classes(y_prob_local, model.classes_, n_classes)
    else:
        global_to_local = {int(global_class): idx for idx, global_class in enumerate(local_classes)}
        y_train_local = np.array([global_to_local[int(y)] for y in y_train], dtype=np.int32)

        model = build_xgboost_model(n_local_classes=len(local_classes))
        model.fit(X_train_flat, y_train_local)

        y_prob_local = model.predict_proba(X_test_flat)
        y_prob = align_proba_to_global_classes(y_prob_local, local_classes, n_classes)
        y_pred = np.argmax(y_prob, axis=1)

    test_acc = accuracy_score(y_test, y_pred)
    test_loss = compute_loss_from_proba(y_test, y_prob, n_classes)

    return {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def train_predict_svm(X_train, y_train, X_test, y_test, n_classes):
    X_train_flat = flatten_windows(X_train)
    X_test_flat = flatten_windows(X_test)

    if len(np.unique(y_train)) == 1:
        model = DummyClassifier(strategy="most_frequent")
    else:
        model = build_svm_model()

    model.fit(X_train_flat, y_train)

    y_pred = model.predict(X_test_flat)
    y_prob_local = model.predict_proba(X_test_flat)
    y_prob = align_proba_to_global_classes(y_prob_local, model.classes_, n_classes)

    test_acc = accuracy_score(y_test, y_pred)
    test_loss = compute_loss_from_proba(y_test, y_prob, n_classes)

    return {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def train_predict_model(model_name, X_train, y_train, X_test, y_test, n_classes):
    if model_name in ["SIMPLE_LSTM","CNN_2LSTM", "STACKED_LSTM", "LSTM"]:
        return train_predict_deep_model(model_name, X_train, y_train, X_test, y_test, n_classes)

    if model_name == "XGBOOST":
        return train_predict_xgboost(X_train, y_train, X_test, y_test, n_classes)

    if model_name == "SVM":
        return train_predict_svm(X_train, y_train, X_test, y_test, n_classes)

    raise ValueError(f"Modelo no reconocido: {model_name}")


# ============================================================
# Guardado de resultados
# ============================================================

def save_global_metrics_for_model(model_name, y_true, y_pred, label_encoder):
    safe_name = safe_model_name(model_name)
    class_names = [str(c) for c in label_encoder.classes_]

    print("\n" + "=" * 70)
    print(f"Classification report global - {model_name}")
    print("=" * 70)

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    print(report_text)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(OUTPUT_DIR, f"classification_report_global_{safe_name}.csv")
    report_df.to_csv(report_path, index=True)
    print(f"Classification report guardado en: {report_path}")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    print("\n" + "=" * 70)
    print(f"Matriz de confusión global - {model_name}")
    print("=" * 70)

    cm_df = pd.DataFrame(
        cm,
        index=[f"Real {c}" for c in class_names],
        columns=[f"Pred {c}" for c in class_names],
    )

    print(cm_df)

    cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_cross_scene_{safe_name}.csv")
    cm_df.to_csv(cm_path)
    print(f"Matriz guardada en: {cm_path}")

    if plt is not None:
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names,
        )

        disp.plot(values_format="d")
        plt.title(f"Matriz de confusión global - {model_name}")
        plt.tight_layout()

        png_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_cross_scene_{safe_name}.png")
        plt.savefig(png_path, dpi=300)
        plt.close()
        print(f"Imagen guardada en: {png_path}")
    else:
        print("matplotlib no está instalado: se omite la imagen PNG de la matriz.")

    return cm_df, report_df


# ============================================================
# Cross-scene validation
# ============================================================

def cross_scene_validation(scene_files):
    ensure_output_dir()
    check_dependencies()

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

    class_names = [str(c) for c in label_encoder.classes_]
    n_classes = len(label_encoder.classes_)

    print("\nClases detectadas:", class_names)
    print("\nCanales usados por el modelo:")
    for col in MODEL_COLS:
        print(f"  - {col}")

    print("\nModelos a ejecutar:")
    for model_name in MODELS_TO_RUN:
        print(f"  - {model_name}")

    global_y_true = {model_name: [] for model_name in MODELS_TO_RUN}
    global_y_pred = {model_name: [] for model_name in MODELS_TO_RUN}

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
            future=FUTURE_SECONDS,
        )

        X_test, y_test, test_discarded, test_total, test_discard_pct = make_centered_windows(
            X_test_scaled,
            y_test_raw,
            anchor_present=anchor_test_present,
            past=PAST_SECONDS,
            future=FUTURE_SECONDS,
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

        for model_name in MODELS_TO_RUN:
            print("\n" + "-" * 70)
            print(f"Modelo: {model_name} | Test scene: {test_scene_name}")
            print("-" * 70)

            result = train_predict_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                n_classes=n_classes,
            )

            test_loss = result["test_loss"]
            test_acc = result["test_acc"]
            y_pred = result["y_pred"]

            print(f"Test loss: {test_loss:.4f}")
            print(f"Test acc : {test_acc:.4f}")

            global_y_true[model_name].extend(y_test)
            global_y_pred[model_name].extend(y_pred)

            fold_results.append({
                "model": model_name,
                "test_scene": test_scene_name,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "n_test_samples": len(y_test),
                "train_windows_total": train_total,
                "train_windows_discarded": train_discarded,
                "train_discard_pct": train_discard_pct,
                "test_windows_total": test_total,
                "test_windows_discarded": test_discarded,
                "test_discard_pct": test_discard_pct,
            })

    # ========================================================
    # Resultados por fold
    # ========================================================

    results_df = pd.DataFrame(fold_results)

    print("\n" + "=" * 70)
    print("Resultados por fold - todos los modelos")
    print("=" * 70)
    print(results_df)

    all_results_path = os.path.join(OUTPUT_DIR, "results_cross_scene_all_models.csv")
    results_df.to_csv(all_results_path, index=False)
    print(f"\nResultados guardados en: {all_results_path}")

    for model_name in MODELS_TO_RUN:
        safe_name = safe_model_name(model_name)
        model_results = results_df[results_df["model"] == model_name].copy()
        model_results_path = os.path.join(OUTPUT_DIR, f"results_cross_scene_{safe_name}.csv")
        model_results.to_csv(model_results_path, index=False)
        print(f"Resultados de {model_name} guardados en: {model_results_path}")

    # ========================================================
    # Resultados globales por modelo
    # ========================================================

    cms = {}
    reports = {}

    for model_name in MODELS_TO_RUN:
        y_true = np.array(global_y_true[model_name])
        y_pred = np.array(global_y_pred[model_name])

        if len(y_true) == 0:
            print(f"\nNo hay predicciones globales para {model_name}.")
            continue

        cm_df, report_df = save_global_metrics_for_model(
            model_name=model_name,
            y_true=y_true,
            y_pred=y_pred,
            label_encoder=label_encoder,
        )

        cms[model_name] = cm_df
        reports[model_name] = report_df

    return results_df, cms, reports, label_encoder


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    results_df, cms, reports, label_encoder = cross_scene_validation(SCENE_FILES)
