from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, pandas_udf
from pyspark.sql.types import *
import pandas as pd
import joblib


scaler = joblib.load("models/scaler_no_rul.pkl")
model = joblib.load("models/rul_regressor_optimized.joblib")
expected_cols = scaler.feature_names_in_

@pandas_udf(DoubleType())
def predict_rul_udf(
    op1: pd.Series, op2: pd.Series, op3: pd.Series,
    s2: pd.Series, s3: pd.Series, s4: pd.Series, s7: pd.Series,
    s8: pd.Series, s9: pd.Series, s11: pd.Series, s12: pd.Series,
    s13: pd.Series, s14: pd.Series, s15: pd.Series, s17: pd.Series,
    s20: pd.Series, s21: pd.Series
) -> pd.Series:
    
    df = pd.DataFrame({
        'op_settings1': op1,
        'op_settings2': op2,
        'op_settings3': op3,
        'sensor2': s2,
        'sensor3': s3,
        'sensor4': s4,
        'sensor7': s7,
        'sensor8': s8,
        'sensor9': s9,
        'sensor11': s11,
        'sensor12': s12,
        'sensor13': s13,
        'sensor14': s14,
        'sensor15': s15,
        'sensor17': s17,
        'sensor20': s20,
        'sensor21': s21,
    })
    
    
    for col_name in ['sensor2', 'sensor3', 'sensor4', 'sensor7', 'sensor8', 'sensor9', 
                     'sensor11', 'sensor12', 'sensor13', 'sensor14', 'sensor15', 
                     'sensor17', 'sensor20', 'sensor21']:
        df[f"{col_name}_mean"] = df[col_name]
        df[f"{col_name}_std"] = 0.0
        
    df = df[expected_cols]
    scaled_data = scaler.transform(df)
    preds = model.predict(scaled_data)
    return pd.Series(preds)

spark = (
    SparkSession.builder
    .appName("Vanguard-Streaming")
    .master("local[*]")
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1"
    )
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .getOrCreate()
)

raw_stream = (
    spark.readStream
    .format("kafka")
    .option(
        "kafka.bootstrap.servers",
        "localhost:9092"
    )
    .option(
        "subscribe",
        "engine-telemetry"
    )
    .option(
        "startingOffsets",
        "latest"
    )
    .load()
)

json_stream = raw_stream.select(
    col("value").cast("string").alias("json_data")
)

schema = StructType([
    StructField("unit_number", IntegerType()),
    StructField("cycle", IntegerType()),
    StructField("op_setting_1", DoubleType()),
    StructField("op_setting_2", DoubleType()),
    StructField("op_setting_3", DoubleType()),
    *[
        StructField(
            f"sensor_{i}",
            DoubleType()
        )
        for i in range(1, 22)
    ]
])

parsed_stream = (
    json_stream
    .select(
        from_json(
            col("json_data"),
            schema
        ).alias("data")
    )
)

telemetry_stream = parsed_stream.select(
    "data.*"
)

telemetry_stream = telemetry_stream.withColumn(
    "predicted_RUL",
    predict_rul_udf(
        col("op_setting_1"), col("op_setting_2"), col("op_setting_3"),
        col("sensor_2"), col("sensor_3"), col("sensor_4"), col("sensor_7"),
        col("sensor_8"), col("sensor_9"), col("sensor_11"), col("sensor_12"),
        col("sensor_13"), col("sensor_14"), col("sensor_15"), col("sensor_17"),
        col("sensor_20"), col("sensor_21")
    )
)

query = (
    telemetry_stream.writeStream
    .format("console")
    .outputMode("append")
    .start()
)

query.awaitTermination()