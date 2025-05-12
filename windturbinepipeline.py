from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev, min, max, avg, count
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.window import Window
import pyspark.sql.functions as F

def load_data(spark: SparkSession, data_path: str) -> DataFrame:
    """
    Load data from CSV files into a DataFrame.

    Args:
        spark (SparkSession): The Spark session object.
        data_path (str): The path to the CSV files to load.

    Returns:
        DataFrame: A PySpark DataFrame containing the loaded data.
    """
    return spark.read.csv(data_path, header=True, inferSchema=True)


def clean_data(df: DataFrame) -> DataFrame:
    """
    Clean the data by handling missing values and removing outliers.

    Args:
        df (DataFrame): The input DataFrame containing raw turbine data.

    Returns:
        DataFrame: A cleaned DataFrame with missing values imputed and outliers removed.
    """
    # Handling missing values (impute with median)
    imputed_df = df.fillna({"wind_speed": df.approxQuantile("wind_speed", [0.5], 0.01)[0],
                            "power_output": df.approxQuantile("power_output", [0.5], 0.01)[0]})

    # Removing outliers using IQR method
    quantiles = df.approxQuantile("power_output", [0.25, 0.75], 0.01)
    Q1, Q3 = quantiles[0], quantiles[1]
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return imputed_df.filter((col("power_output") >= lower_bound) & (col("power_output") <= upper_bound))


def compute_summary_statistics(df: DataFrame) -> DataFrame:
    """
    Compute summary statistics (min, max, avg, and count) for each turbine.

    Args:
        df (DataFrame): The cleaned DataFrame containing turbine data.

    Returns:
        DataFrame: A DataFrame with summary statistics for each turbine.
    """
    return df.groupBy("turbine_id").agg(
        min("power_output").alias("min_power_output"),
        max("power_output").alias("max_power_output"),
        avg("power_output").alias("avg_power_output"),
        count("power_output").alias("record_count")
    )


def detect_anomalies(df: DataFrame) -> DataFrame:
    """
    Detect anomalies in turbine power output by identifying values outside of 2 standard deviations from the mean.

    Args:
        df (DataFrame): The cleaned DataFrame containing turbine data.

    Returns:
        DataFrame: A DataFrame containing the turbines with anomalous power output.
    """
    window_spec = Window.partitionBy("turbine_id")
    df = df.withColumn("mean_power", mean("power_output").over(window_spec))
    df = df.withColumn("stddev_power", stddev("power_output").over(window_spec))
    df = df.withColumn("upper_bound", col("mean_power") + 2 * col("stddev_power"))
    df = df.withColumn("lower_bound", col("mean_power") - 2 * col("stddev_power"))
    return df.filter((col("power_output") > col("upper_bound")) | (col("power_output") < col("lower_bound")))


def save_to_parquet(df: DataFrame, output_path: str) -> None:
    """
    Save the processed DataFrame to a Parquet file.

    Args:
        df (DataFrame): The DataFrame to save.
        output_path (str): The output file path to save the Parquet file.
    """
    df.write.mode("overwrite").parquet(output_path)


def main() -> None:
    """
    Main function to orchestrate the data pipeline.
    1. Loads data.
    2. Cleans the data.
    3. Computes summary statistics.
    4. Detects anomalies.
    5. Saves the cleaned data, summary statistics, and anomalies to Parquet files.
    """
    spark = SparkSession.builder.master("local").appName("WindTurbine").getOrCreate()

    data_path = "C:\ARUN\PythonLabs\calibri\*.csv"
    output_path = "C:\ARUN\PythonLabs\calibri"

    df = load_data(spark, data_path)
    cleaned_df = clean_data(df)
    summary_df = compute_summary_statistics(cleaned_df)
    anomalies_df = detect_anomalies(cleaned_df)

    save_to_parquet(cleaned_df, f"{output_path}\cleaned_data")
    save_to_parquet(summary_df, f"{output_path}\summary_statistics")
    save_to_parquet(anomalies_df, f"{output_path}\\anomalies")

    spark.stop()


if __name__ == "__main__":
    main()
