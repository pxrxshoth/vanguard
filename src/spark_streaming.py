"""
spark_streaming.py - Vanguard PySpark Structured Streaming Job

Pipeline:
  Kafka (engine-telemetry) → PySpark → RUL Prediction + Anomaly Detection
  → POST /api/v1/telemetry → FastAPI → WebSocket → React Dashboard

Run from the repository root:
    python src/spark_streaming.py
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType, StructField, StructType,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vanguard.spark")

# ---------------------------------------------------------------------------
# Paths & configuration
# ---------------------------------------------------------------------------

BASE_PATH = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_PATH / "models"
API_INGEST_URL = "http://localhost:8000/api/v1/telemetry"

# ---------------------------------------------------------------------------
# Load models once in the driver — they are broadcast into Pandas UDFs
# ---------------------------------------------------------------------------

scaler = joblib.load(MODELS_DIR / "scaler_no_rul.pkl")
rul_model = joblib.load(MODELS_DIR / "rul_regressor_optimized.joblib")
anomaly_model = joblib.load(MODELS_DIR / "anomaly_detector_if.joblib")

EXPECTED_COLS = scaler.feature_names_in_.tolist()

# Isolation Forest decision_function: more negative → more anomalous.
# We normalise it to [0, 1] where 1 = definitely anomaly.
IF_THRESHOLD = 0.0   # IF score < 0 means labelled as outlier by sklearn

SENSOR_FEATURE_COLS = [
    "sensor2", "sensor3", "sensor4", "sensor7",
    "sensor8", "sensor9", "sensor11", "sensor12",
    "sensor13", "sensor14", "sensor15", "sensor17",
    "sensor20", "sensor21",
]

# ---------------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------------

spark = (
    SparkSession.builder
    .appName("Vanguard-Streaming")
    .master("local[*]")
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1",
    )
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# Kafka schema
# ---------------------------------------------------------------------------

SCHEMA = StructType([
    StructField("unit_number", IntegerType()),
    StructField("cycle", IntegerType()),
    StructField("op_setting_1", DoubleType()),
    StructField("op_setting_2", DoubleType()),
    StructField("op_setting_3", DoubleType()),
    *[StructField(f"sensor_{i}", DoubleType()) for i in range(1, 22)],
])

# ---------------------------------------------------------------------------
# Read from Kafka
# ---------------------------------------------------------------------------

raw_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", "engine-telemetry")
    .option("startingOffsets", "latest")
    .load()
)

telemetry_stream = (
    raw_stream
    .select(col("value").cast(StringType()).alias("json_data"))
    .select(from_json(col("json_data"), SCHEMA).alias("data"))
    .select("data.*")
)


# ---------------------------------------------------------------------------
# foreachBatch sink — run inference and POST to FastAPI
# ---------------------------------------------------------------------------

def build_feature_df(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix that the scaler/RUL model expects.
    Mirrors the feature engineering done during training:
        - Raw sensor values
        - Per-sensor rolling mean and std (approximated as single-row stubs)
    """
    rename_map = {
        "op_setting_1": "op_settings1",
        "op_setting_2": "op_settings2",
        "op_setting_3": "op_settings3",
    }
    for i in range(1, 22):
        rename_map[f"sensor_{i}"] = f"sensor{i}"
    pdf = pdf.rename(columns=rename_map)

    # Rolling features: for a streaming single-row context, mean == value and std == 0
    for s in SENSOR_FEATURE_COLS:
        pdf[f"{s}_roll_mean"] = pdf[s]
        pdf[f"{s}_roll_std"] = 0.0

    # Align to training column order, fill any missing with 0
    pdf = pdf.reindex(columns=EXPECTED_COLS, fill_value=0.0)
    return pdf


def process_batch(batch_df: DataFrame, batch_id: int) -> None:
    """
    Called by Spark for each micro-batch.
    Runs RUL regression and anomaly detection, then POSTs results to FastAPI.
    """
    pdf = batch_df.toPandas()
    if pdf.empty:
        return

    logger.info("[batch %d] Processing %d rows ...", batch_id, len(pdf))

    feature_df = build_feature_df(pdf.copy())
    scaled = scaler.transform(feature_df)

    # --- RUL prediction ---
    rul_preds = np.clip(rul_model.predict(scaled), 0, 125)

    # --- Anomaly detection ---
    # decision_function returns negative scores for outliers
    if_scores = anomaly_model.decision_function(feature_df)
    # Normalise to [0, 1]: score = 0 → borderline, score = -1 → strong outlier
    anomaly_scores = np.clip(-if_scores, 0, 1)
    is_anomaly_flags = anomaly_model.predict(feature_df) == -1  # -1 == outlier

    # --- POST to FastAPI ---
    for idx, row in pdf.iterrows():
        i = pdf.index.get_loc(idx)
        payload = {
            "unit_number": int(row["unit_number"]),
            "cycle": int(row["cycle"]),
            "predicted_rul": float(rul_preds[i]),
            "anomaly_score": float(anomaly_scores[i]),
            "is_anomaly": bool(is_anomaly_flags[i]),
        }
        try:
            resp = requests.post(API_INGEST_URL, json=payload, timeout=2)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("[batch %d] Failed to POST unit %d: %s", batch_id, payload["unit_number"], exc)

    logger.info("[batch %d] Done.", batch_id)


# ---------------------------------------------------------------------------
# Start the streaming query
# ---------------------------------------------------------------------------

query = (
    telemetry_stream.writeStream
    .foreachBatch(process_batch)
    .outputMode("append")
    .option("checkpointLocation", str(BASE_PATH / "data/.spark_checkpoint"))
    .start()
)

logger.info("Vanguard Spark Streaming job started. Waiting for Kafka messages...")
query.awaitTermination()