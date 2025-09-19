import re
from pyspark.sql import functions as F

RESERVED_KEYWORDS = {"case", "select", "from", "table"}

def sanitize_name(name: str) -> str:
    """Sanitize column names and handle reserved words."""
    sanitized = re.sub(r"[ ,;{}()\n\t=]", "_", name)
    if sanitized.lower() in RESERVED_KEYWORDS:
        sanitized = sanitized + "_col"
    return sanitized

def add_ingestion_columns(df):
    """Adds ingestion metadata columns for SCD2 / traceability."""
    return (df
            #.withColumn("ingested_at", F.current_timestamp())
            .withColumn("is_current", F.lit(True)))
    
# def read_bronze_table(spark, catalog, schema, table):
#     return spark.table(f"{catalog}.{schema}.{table}")

# def write_silver_table(df, catalog, schema, table, mode="overwrite"):
#     df.write.format("delta").mode(mode).saveAsTable(f"{catalog}.{schema}.{table}")

# def add_ingested_at(df):
#     from pyspark.sql.functions import current_timestamp
#     return df.withColumn("ingested_at", current_timestamp())

# def scd2_merge(spark, df_new, target_table, key_columns, watermark_col):
#     # merge logic for SCD2
#     pass


