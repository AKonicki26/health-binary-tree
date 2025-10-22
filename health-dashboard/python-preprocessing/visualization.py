import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def load_data(data_path):
    """Load cleaned data from CSV"""
    print("=== Loading Data ===")
    data = pd.read_csv(data_path)
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    print(f"Total records: {len(data)}")
    return data


def prepare_blood_pressure_data(data):
    """Prepare blood pressure data for analysis"""
    print("\n=== Preparing Blood Pressure Data ===")
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
    
    print(f"✓ Data binned into {len(blood_pressure_labels)} ranges")
    return bp_data, blood_pressure_labels


def calculate_bp_statistics(bp_data, blood_pressure_labels):
    """Calculate COVID statistics by blood pressure range"""
    print("\n=== Calculating Statistics ===")
    
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
    
    print("✓ Statistics calculated for all ranges")
    return total_counts, positive_counts, negative_counts, positive_percent, negative_percent


def create_bp_visualization(blood_pressure_labels, positive_percent, negative_percent, output_path):
    """Create and save blood pressure visualization"""
    print("\n=== Creating Visualization ===")
    
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Chart saved to: {output_path}")
    
    plt.close()


def print_bp_summary(bp_data, blood_pressure_labels, positive_percent):
    """Print summary statistics"""
    print("\n=== Blood Pressure Analysis Summary ===")
    print(f"Total records analyzed: {len(bp_data)}")
    print(f"\nCOVID Positive Rate by Blood Pressure Range:")
    for label, pct in zip(blood_pressure_labels, positive_percent):
        print(f"  {label}: {pct:.1%}")


def create_blood_pressure_visualization(data):
    """Create blood pressure visualization - orchestrates all BP-related functions"""
    print("\n" + "="*60)
    print("BLOOD PRESSURE VISUALIZATION")
    print("="*60)
    
    # Prepare blood pressure data
    bp_data, blood_pressure_labels = prepare_blood_pressure_data(data)
    
    # Calculate statistics
    total_counts, positive_counts, negative_counts, positive_percent, negative_percent = calculate_bp_statistics(
        bp_data, blood_pressure_labels
    )
    
    # Create visualization
    output_path = "../public/bp-analysis.png"
    create_bp_visualization(blood_pressure_labels, positive_percent, negative_percent, output_path)
    
    # Print summary
    print_bp_summary(bp_data, blood_pressure_labels, positive_percent)
    
    print("\n✓ Blood Pressure Visualization Complete!")

def prepare_heart_rate_data(data):
    """Prepare heart rate data for analysis"""
    print("\n=== Preparing Heart Rate Data ===")

    # Create heart rate data
    hr_data = data[["Heart_Rate", "Is_Covid_True"]].copy()

    # Define heart rate bins and labels,
    min_heart_rate = 60
    max_heart_rate = 120
    step = 10
    heart_rate_ranges = range(min_heart_rate, max_heart_rate + step, step)
    heart_rate_labels = [f'{i}-{i + step}' for i in heart_rate_ranges[:-1]]

    # Bin the data,
    hr_data['hr_range'] = pd.cut(
        hr_data['Heart_Rate'],
        bins=heart_rate_ranges,
        labels=heart_rate_labels,
        right=False
    )

    print(f"✓ Data binned into {len(heart_rate_labels)} ranges")
    return hr_data, heart_rate_labels


def calculate_hr_statistics(hr_data, heart_rate_labels):
    """Calculate COVID statistics by heart rate range"""
    print("\n=== Calculating Statistics ===")

    # Count total, positive, and negative entries
    total_counts = hr_data['hr_range'].value_counts().sort_index()
    positive_counts = hr_data[hr_data['Is_Covid_True'] == 1]['hr_range'].value_counts().sort_index()
    negative_counts = total_counts - positive_counts

    # Reindex to include all bins
    total_counts = total_counts.reindex(heart_rate_labels, fill_value=0)
    positive_counts = positive_counts.reindex(heart_rate_labels, fill_value=0)
    negative_counts = negative_counts.reindex(heart_rate_labels, fill_value=0)

    # Avoid division by zero
    positive_percent = (positive_counts / total_counts).fillna(0)
    negative_percent = (negative_counts / total_counts).fillna(0)

    print("✓ Statistics calculated for all ranges")
    return total_counts, positive_counts, negative_counts, positive_percent, negative_percent


def create_hr_visualization(heart_rate_labels, positive_percent, negative_percent, output_path):
    """Create and save heart rate visualization"""
    print("\n=== Creating Visualization ===")

    # Plot 100% stacked bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(heart_rate_labels, negative_percent, bottom=positive_percent,
            label='COVID Negative', color='#ef4444')
    plt.bar(heart_rate_labels, positive_percent,
            label='COVID Positive', color='#22c55e')

    plt.title("COVID-19 Cases by Heart Rate Range", fontsize=12, fontweight='bold')
    plt.xlabel("Heart Rate Range", fontsize=14)
    plt.ylabel("Percentage", fontsize=12)
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    # Save the plot to the public folder
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Chart saved to: {output_path}")

    plt.close()


def print_hr_summary(hr_data, heart_rate_labels, positive_percent):
    """Print summary statistics"""
    print("\n=== Blood Pressure Analysis Summary ===")
    print(f"Total records analyzed: {len(hr_data)}")
    print(f"\nCOVID Positive Rate by Blood Pressure Range:")
    for label, pct in zip(heart_rate_labels, positive_percent):
        print(f"  {label}: {pct:.1%}")


def create_heart_rate_visualization(data):
    """Create blood pressure visualization - orchestrates all HR-related functions"""
    print("\n" + "=" * 60)
    print("HEART RATE VISUALIZATION")
    print("=" * 60)

    # Prepare blood pressure data
    hr_data, heart_rate_labels = prepare_heart_rate_data(data)

    # Calculate statistics
    total_counts, positive_counts, negative_counts, positive_percent, negative_percent = calculate_hr_statistics(
        hr_data, heart_rate_labels
    )

    # Create visualization
    output_path = "../public/hr-analysis.png"
    create_bp_visualization(heart_rate_labels, positive_percent, negative_percent, output_path)

    # Print summary
    print_bp_summary(hr_data, heart_rate_labels, positive_percent)

    print("\n✓ Heart Rate Visualization Complete!")


def prepare_comorbidity_data(data):
    """Prepare comorbidity data for analysis"""
    print("\n=== Preparing Comorbidity Data ===")

    # Create comorbidity data
    cmbd_data = data[["Comorbidity_Count", "Is_Covid_True"]].copy()

    # Define comorbidity labels
    comorbidity_labels = [0, 1, 2, 3, 4]

    return cmbd_data, comorbidity_labels


def calculate_cmbd_statistics(cmbd_data, comorbidity_labels):
    """Calculate COVID statistics by comorbidity range"""
    print("\n=== Calculating Statistics ===")

    # Count total, positive, and negative entries
    total_counts = cmbd_data['cmbd_range'].value_counts().sort_index()
    positive_counts = cmbd_data[cmbd_data['Is_Covid_True'] == 1]['cmbd_range'].value_counts().sort_index()
    negative_counts = total_counts - positive_counts

    # Reindex to include all bins
    total_counts = total_counts.reindex(comorbidity_labels, fill_value=0)
    positive_counts = positive_counts.reindex(comorbidity_labels, fill_value=0)
    negative_counts = negative_counts.reindex(comorbidity_labels, fill_value=0)

    # Avoid division by zero
    positive_percent = (positive_counts / total_counts).fillna(0)
    negative_percent = (negative_counts / total_counts).fillna(0)

    print("✓ Statistics calculated for all ranges")
    return total_counts, positive_counts, negative_counts, positive_percent, negative_percent


def create_cmbd_visualization(comorbidity_labels, positive_percent, negative_percent, output_path):
    """Create and save comorbidity visualization"""
    print("\n=== Creating Visualization ===")

    # Plot 100% stacked bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(comorbidity_labels, negative_percent, bottom=positive_percent,
            label='COVID Negative', color='#ef4444')
    plt.bar(comorbidity_labels, positive_percent,
            label='COVID Positive', color='#22c55e')

    plt.title("COVID-19 Cases by Comorbidity Range", fontsize=12, fontweight='bold')
    plt.xlabel("Comorbidity Range", fontsize=14)
    plt.ylabel("Percentage", fontsize=12)
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    # Save the plot to the public folder
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Chart saved to: {output_path}")

    plt.close()


def print_cmbd_summary(cmbd_data, comorbidity_labels, positive_percent):
    """Print summary statistics"""
    print("\n=== Comorbidity Analysis Summary ===")
    print(f"Total records analyzed: {len(cmbd_data)}")
    print(f"\nCOVID Positive Rate by Comorbidity:")
    for label, pct in zip(comorbidity_labels, positive_percent):
        print(f"  {label}: {pct:.1%}")


def create_comorbidity_visualization(data):
    """Create comorbidity visualization - orchestrates all comorbidity-related functions"""
    print("\n" + "=" * 60)
    print("COMORBIDITY VISUALIZATION")
    print("=" * 60)

    # Prepare comorbidity data
    cmbd_data, comorbidity_labels = prepare_comorbidity_data(data)

    # Calculate statistics
    total_counts, positive_counts, negative_counts, positive_percent, negative_percent = calculate_cmbd_statistics(
        cmbd_data, comorbidity_labels
    )

    # Create visualization
    output_path = "../public/cmbd-analysis.png"
    create_cmbd_visualization(comorbidity_labels, positive_percent, negative_percent, output_path)

    # Print summary
    print_cmbd_summary(cmbd_data, comorbidity_labels, positive_percent)

    print("\n✓ Comorbidity Visualization Complete!")

def create_age_visualization(data):
    """Create age visualization - orchestrates all age-related functions"""
    print("\n" + "=" * 60)
    print("AGE VISUALIZATION")
    print("=" * 60)

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

    # === Age ANALYSIS ===
    print("\n=== Age Analysis ===")

    # Create blood pressure data
    cmbd_data = data[["Age", "Is_Covid_True"]].copy()

    plt.figure(figsize=(10, 6))
    plt.hist(cmbd_data["Age"], bins=100)

    plt.title("Age Distribution of Covid-19 Survey Respondants", fontsize=14, fontweight='bold')
    plt.xlabel("Person Count", fontsize=12)
    plt.ylabel("Age", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    # Save the plot to the public folder
    output_path = "./health-dashboard/public/age-analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Chart saved to: {output_path}")


def main():
    """Main visualization pipeline"""
    # Load data once
    data_path = "../public/input.csv"
    data = load_data(data_path)
    
    # Create blood pressure visualization
    create_blood_pressure_visualization(data)

    # Create heart rate visualization
    create_heart_rate_visualization(data)

    # Create comorbidity visualization
    create_comorbidity_visualization(data)

    # Create age visualization
    create_age_visualization(data)

    # Future visualizations can be added here:
    # create_age_visualization(data)
    # create_symptom_visualization(data)
    # create_comorbidity_visualization(data)
    
    print("\n" + "="*60)
    print("✓ ALL VISUALIZATIONS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()