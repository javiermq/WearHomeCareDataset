"""Statistical analysis of IMU variables in a long-format sensor dataset.

Expected input columns:

    timestamp,sensor,label,valor

The script aggregates sensor samples by second using the median, assigns the
modal label to each second, and exports:

1. Statistics per label and variable: count, mean, standard deviation, median,
   minimum and maximum.
2. Pearson correlation matrix between IMU variables.
3. Pearson correlation matrices calculated separately for each label.
4. Correlation between each variable and the label when exactly two labels
   are present (point-biserial correlation, equivalent to Pearson with 0/1).
5. The processed wide dataset, with one row per second.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd


DEFAULT_IMU_SENSORS = [
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

REQUIRED_COLUMNS = ["timestamp", "sensor", "label", "valor"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate mean, standard deviation, median and correlations "
            "for MAG and the remaining IMU variables."
        )
    )
    parser.add_argument(
        "dataset",
        help="Input CSV with timestamp,sensor,label,valor columns.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Default: imu_analysis_<input filename> next "
            "to the input dataset."
        ),
    )
    parser.add_argument(
        "--sensors",
        nargs="+",
        default=DEFAULT_IMU_SENSORS,
        help="Sensor variables to analyse. Default: all known IMU variables.",
    )
    parser.add_argument(
        "--time-unit",
        choices=["second", "timestamp"],
        default="second",
        help=(
            "Aggregate samples by relative second or exact timestamp. "
            "Default: second."
        ),
    )
    return parser.parse_args()


def safe_filename(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def modal_label(values):
    modes = values.mode()
    if modes.empty:
        return values.iloc[0]
    return modes.iloc[0]


def load_and_transform(dataset_path, sensors, time_unit):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"No existe el dataset: {dataset_path}")

    raw = pd.read_csv(dataset_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas en {dataset_path}: {missing}"
        )

    raw = raw[REQUIRED_COLUMNS].copy()
    raw["timestamp"] = pd.to_numeric(raw["timestamp"], errors="coerce")
    raw["valor"] = pd.to_numeric(raw["valor"], errors="coerce")
    raw = raw.dropna(subset=REQUIRED_COLUMNS)
    raw = raw[raw["sensor"].isin(sensors)]

    if raw.empty:
        raise ValueError(
            "No hay muestras para los sensores solicitados en el dataset."
        )

    available_sensors = [
        sensor for sensor in sensors if sensor in set(raw["sensor"])
    ]
    missing_sensors = [
        sensor for sensor in sensors if sensor not in set(raw["sensor"])
    ]

    if time_unit == "second":
        raw["time_index"] = (
            (raw["timestamp"] - raw["timestamp"].min()) // 1000
        ).astype(np.int64)
        time_column = "second"
    else:
        raw["time_index"] = raw["timestamp"].astype(np.int64)
        time_column = "timestamp"

    labels = (
        raw.groupby("time_index")["label"]
        .agg(modal_label)
        .rename("label")
    )
    values = raw.pivot_table(
        index="time_index",
        columns="sensor",
        values="valor",
        aggfunc="median",
    )

    processed = labels.to_frame().join(values, how="inner").reset_index()
    processed = processed.rename(columns={"time_index": time_column})
    processed = processed[[time_column, "label"] + available_sensors]

    # Correlations require paired observations. Interpolation is deliberately
    # avoided: missing measurements remain NaN and pandas uses pairwise rows.
    return processed, available_sensors, missing_sensors


def calculate_statistics(processed, sensors):
    rows = []
    for label, label_data in processed.groupby("label", sort=True):
        for sensor in sensors:
            values = label_data[sensor].dropna()
            rows.append({
                "label": label,
                "sensor": sensor,
                "count": int(values.count()),
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "median": values.median(),
                "min": values.min(),
                "max": values.max(),
            })

    all_labels = processed["label"].nunique()
    for sensor in sensors:
        values = processed[sensor].dropna()
        rows.append({
            "label": "ALL",
            "sensor": sensor,
            "count": int(values.count()),
            "mean": values.mean(),
            "std": values.std(ddof=1),
            "median": values.median(),
            "min": values.min(),
            "max": values.max(),
        })

    statistics = pd.DataFrame(rows)
    statistics.attrs["n_labels"] = all_labels
    return statistics


def calculate_label_correlations(processed, sensors):
    labels = sorted(processed["label"].astype(str).unique())
    if len(labels) != 2:
        return pd.DataFrame([{
            "sensor": sensor,
            "correlation_with_label": np.nan,
            "label_0": "",
            "label_1": "",
            "note": (
                f"No calculada: hay {len(labels)} etiquetas. "
                "La correlación con una etiqueta numérica solo es directamente "
                "interpretable para dos clases."
            ),
        } for sensor in sensors])

    encoded = processed["label"].astype(str).map({
        labels[0]: 0,
        labels[1]: 1,
    })
    rows = []
    for sensor in sensors:
        paired = pd.concat(
            [processed[sensor], encoded.rename("encoded_label")],
            axis=1,
        ).dropna()
        rows.append({
            "sensor": sensor,
            "correlation_with_label": paired[sensor].corr(
                paired["encoded_label"]
            ),
            "label_0": labels[0],
            "label_1": labels[1],
            "note": (
                f"Pearson/point-biserial: {labels[0]}=0, {labels[1]}=1"
            ),
        })
    return pd.DataFrame(rows)


def save_results(
    processed,
    sensors,
    missing_sensors,
    dataset_path,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    statistics = calculate_statistics(processed, sensors)
    correlations = processed[sensors].corr(method="pearson")
    label_correlations = calculate_label_correlations(processed, sensors)

    statistics.to_csv(
        os.path.join(output_dir, "imu_statistics_by_label.csv"),
        index=False,
    )
    correlations.to_csv(
        os.path.join(output_dir, "imu_correlation_matrix_all.csv"),
    )
    label_correlations.to_csv(
        os.path.join(output_dir, "imu_correlation_with_label.csv"),
        index=False,
    )
    processed.to_csv(
        os.path.join(output_dir, "imu_processed_by_time.csv"),
        index=False,
    )

    for label, label_data in processed.groupby("label", sort=True):
        label_data[sensors].corr(method="pearson").to_csv(
            os.path.join(
                output_dir,
                f"imu_correlation_matrix_label_{safe_filename(label)}.csv",
            )
        )

    metadata = pd.DataFrame([
        {"field": "dataset", "value": os.path.abspath(dataset_path)},
        {"field": "processed_rows", "value": len(processed)},
        {
            "field": "labels",
            "value": ", ".join(map(str, sorted(processed["label"].unique()))),
        },
        {"field": "analysed_sensors", "value": ", ".join(sensors)},
        {"field": "missing_sensors", "value": ", ".join(missing_sensors)},
        {
            "field": "aggregation",
            "value": "Median per sensor and time index; modal label",
        },
    ])
    metadata.to_csv(
        os.path.join(output_dir, "analysis_metadata.csv"),
        index=False,
    )

    return statistics, label_correlations


def main():
    args = parse_args()
    dataset_path = os.path.abspath(args.dataset)

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        output_dir = os.path.join(
            os.path.dirname(dataset_path),
            f"imu_analysis_{safe_filename(dataset_name)}",
        )

    processed, sensors, missing_sensors = load_and_transform(
        dataset_path=dataset_path,
        sensors=args.sensors,
        time_unit=args.time_unit,
    )
    statistics, label_correlations = save_results(
        processed=processed,
        sensors=sensors,
        missing_sensors=missing_sensors,
        dataset_path=dataset_path,
        output_dir=output_dir,
    )

    print(f"Dataset: {dataset_path}")
    print(f"Filas procesadas: {len(processed)}")
    print(f"Etiquetas: {sorted(processed['label'].unique().tolist())}")
    print(f"Sensores analizados: {sensors}")
    if missing_sensors:
        print(f"Sensores ausentes: {missing_sensors}")
    print(f"Resultados guardados en: {output_dir}")

    print("\nMedia, desviación y mediana por etiqueta:")
    print(
        statistics[
            statistics["label"].astype(str) != "ALL"
        ][["label", "sensor", "mean", "std", "median"]].to_string(index=False)
    )

    if processed["label"].nunique() == 2:
        print("\nCorrelación con la etiqueta:")
        print(
            label_correlations[
                ["sensor", "correlation_with_label"]
            ].sort_values(
                "correlation_with_label",
                key=lambda values: values.abs(),
                ascending=False,
            ).to_string(index=False)
        )


if __name__ == "__main__":
    main()
