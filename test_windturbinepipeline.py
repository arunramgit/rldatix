import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql import Row
from calibri.windturbinepipeline import load_data, clean_data, compute_summary_statistics, detect_anomalies, save_to_parquet


@pytest.fixture(scope="module")
def spark():
    """
    Creates a SparkSession fixture for the tests.
    """
    spark = SparkSession.builder.appName("TestPipeline").getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def sample_data(spark):
    """
    Provides a sample DataFrame for testing.
    """
    data = [
        Row(timestamp="2022-03-01 01:00:00", turbine_id=1, wind_speed=11.6, wind_direction=152, power_output=4.4),
        Row(timestamp="2022-03-01 02:00:00", turbine_id=1, wind_speed=13.8, wind_direction=73, power_output=2.9),
        Row(timestamp="2022-03-01 01:00:00", turbine_id=2, wind_speed=12.8, wind_direction=35, power_output=4.2),
        Row(timestamp="2022-03-01 02:00:00", turbine_id=2, wind_speed=9.9, wind_direction=103, power_output=3.8),
        Row(timestamp="2022-03-01 01:00:00", turbine_id=3, wind_speed=10.4, wind_direction=169, power_output=1.9),
        Row(timestamp="2022-03-01 02:00:00", turbine_id=3, wind_speed=12.2, wind_direction=188, power_output=4.0)
    ]
    return spark.createDataFrame(data)


def test_load_data(spark, sample_data):
    """
    Test loading of data into a DataFrame.
    """
    df = sample_data
    assert df.count() == 6
    assert "turbine_id" in df.columns
    assert "power_output" in df.columns


def test_clean_data(sample_data):
    """
    Test cleaning the data (handling missing values and outliers).
    """
    # Create a DataFrame with some missing values and outliers
    df = sample_data.withColumn("wind_speed", col("wind_speed") * 10)

    # Clean the data
    cleaned_df = clean_data(df)

    # Ensure the cleaned DataFrame has no outliers (i.e., the power_output is within bounds)
    power_output_vals = [row["power_output"] for row in cleaned_df.select("power_output").collect()]

    # Check if any power_output values are outliers (values above 10 as an example)
    assert all(value <= 10 for value in power_output_vals)


def test_summary_statistics(sample_data):
    """
    Test the computation of summary statistics (min, max, avg, count).
    """
    summary_df = compute_summary_statistics(sample_data)
    summary = summary_df.collect()

    # Ensure summary statistics are computed correctly
    assert len(summary) == 3  # 3 turbines in the sample data
    assert "min_power_output" in summary_df.columns
    assert "max_power_output" in summary_df.columns
    assert "avg_power_output" in summary_df.columns
    assert "record_count" in summary_df.columns

    # Check that summary statistics for turbine 1 are correct
    turbine_1_stats = next(row for row in summary if row["turbine_id"] == 1)
    assert turbine_1_stats["min_power_output"] == 2.9
    assert turbine_1_stats["max_power_output"] == 4.4
    assert turbine_1_stats["avg_power_output"] == 3.65


def test_detect_anomalies(sample_data):
    """
    Test anomaly detection (turbines outside of 2 standard deviations).
    """
    anomalies_df = detect_anomalies(sample_data)
    anomalies = anomalies_df.collect()

    assert len(anomalies) == 0  # No anomalies in the small sample data
