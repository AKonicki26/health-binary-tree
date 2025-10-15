import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_csv_location = "public/input.csv"
data = pd.read_csv(data_csv_location)
target_col = "Is_Covid_True"

data = pd.read_csv(data_csv_location)

# Make binary columns based on gender
data["Is_Male"] = (data["Gender"] == "Male").astype(int)
data["Is_Female"] = (data["Gender"] == "Female").astype(int)
data["Gender_Other"] = (data["Gender"] == "Other").astype(int)

data = data.drop(["Patient_ID", "Gender", "Name"], axis=1)

data.describe()

# Print concise information about the DataFrame: column names, non-null counts, and data types
data.info()

# Count the number of missing (NaN) values in each column
data.isnull().sum()

# create blood pressure data
bp_data = data[["Blood_Pressure", "Is_Covid_True"]]

# Define blood pressure bins and labels
min_blood_pressure = 80
max_blood_pressure = 180
step = 10
blood_pressure_ranges = range(min_blood_pressure, max_blood_pressure + step, step)
blood_pressure_labels = [f'{i}-{i+step}' for i in blood_pressure_ranges[:-1]]

# Bin the data
bp_data['bp_range'] = pd.cut(
    bp_data['Blood_Pressure'],
    bins=blood_pressure_ranges,
    labels=blood_pressure_labels,
    right=False
)

# Count total, positive, and negative entries
total_counts = bp_data['bp_range'].value_counts().sort_index()
positive_counts = bp_data[bp_data['Is_Covid_True'] == 1]['bp_range'].value_counts().sort_index()
negative_counts = total_counts - positive_counts

# Reindex to include all bins
total_counts = total_counts.reindex(blood_pressure_labels, fill_value=0)
positive_counts = positive_counts.reindex(blood_pressure_labels, fill_value=0)
negative_counts = negative_counts.reindex(blood_pressure_labels, fill_value=0)

# Avoid division by zero (e.g., empty bins)
positive_percent = (positive_counts / total_counts).fillna(0)
negative_percent = (negative_counts / total_counts).fillna(0)

# Plot 100% stacked bar chart
plt.figure(figsize=(8, 6))
plt.bar(blood_pressure_labels, negative_percent, bottom=positive_percent, label='COVID Negative', color='red')
plt.bar(blood_pressure_labels, positive_percent, label='COVID Positive', color='green')


plt.title("Percentage of COVID-19 Cases by Blood Pressure Range")
plt.xlabel("Blood Pressure Range")
plt.ylabel("Percentage")
plt.ylim(0, 1)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()