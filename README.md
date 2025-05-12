**SOLUTION DESIGN:-**


**Data Ingestion:**

Load multiple CSV files from a given directory.


Infer schema and ensure consistent data types.


Handle missing data and corrupt records.


**Data Cleaning:**


Impute missing values using median values.


Remove outliers based on the interquartile range (IQR).


**Summary Statistics Calculation:**


Compute the minimum, maximum, and average power output per turbine over a 24-hour period.


**Anomaly Detection:**


Calculate the mean and standard deviation of power output for each turbine.


Identify anomalies where power output deviates beyond 2 standard deviations from the mean.


**Data Storage:**


Store cleaned data and computed statistics in Parquet files.





**ASSUMPTIONS:-**

Timestamp format is consistent across all files.

Sensor issues can lead to missing records but not duplicated ones.
