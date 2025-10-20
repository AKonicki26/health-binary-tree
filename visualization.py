import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the CLEANED data (already processed by the notebook)
data_csv_location = "./input.csv"
data = pd.read_csv(data_csv_location)

print("=== Data loaded from cleaned input.csv ===")
print(f"Shape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# No need to clean again - data is already cleaned!
# The notebook already:
# - Dropped NaN values
# - Created Is_Male, Is_Female, Gender_Other columns
# - Replaced False/True with 0/1
# - Dropped Patient_ID, Gender, Name columns

# Verify data is clean
print(f"\nMissing values: {data.isnull().sum().sum()}")
print(f"Total records: {len(data)}")

# === BLOOD PRESSURE ANALYSIS ===
print("\n=== Generating Blood Pressure Analysis ===")

# Create blood pressure data
bp_data = data[["Blood_Pressure", "Is_Covid_True"]].copy()

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

# Avoid division by zero
positive_percent = (positive_counts / total_counts).fillna(0)
negative_percent = (negative_counts / total_counts).fillna(0)

# Plot 100% stacked bar chart
plt.figure(figsize=(10, 6))
plt.bar(blood_pressure_labels, negative_percent, bottom=positive_percent, 
        label='COVID Negative', color='#ef4444')
plt.bar(blood_pressure_labels, positive_percent, 
        label='COVID Positive', color='#22c55e')

plt.title("COVID-19 Cases by Blood Pressure Range", fontsize=14, fontweight='bold')
plt.xlabel("Blood Pressure Range (mmHg)", fontsize=12)
plt.ylabel("Percentage", fontsize=12)
plt.ylim(0, 1)
plt.legend(loc='best')
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()

# Save the plot to the public folder
output_path = "./health-dashboard/public/bp-analysis.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n✓ Chart saved to: {output_path}")

plt.close()

# Print summary statistics
print("\n=== Blood Pressure Analysis Summary ===")
print(f"Total records analyzed: {len(bp_data)}")
print(f"\nCOVID Positive Rate by Blood Pressure Range:")
for label, pct in zip(blood_pressure_labels, positive_percent):
    print(f"  {label}: {pct:.1%}")

print("\n✓ Visualization complete!")