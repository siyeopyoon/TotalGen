import csv
import numpy as np
import math
import itertools
from scipy.stats import f_oneway, ttest_ind
import matplotlib.pyplot as plt
import torch

if torch.cuda.is_available():
    use_gpu=True
    external="/external"
else:
    use_gpu=False
    external="/Volumes/HOMEDIR$"
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

def compute_average_std(values):
    # Return average and standard deviation for a list of numbers.
    return np.mean(values), np.std(values)

# List of files to process.
files = [
    #"evaluation_log_Autopet.txt",
    "cascade-Flow-3D-img224","cascade-EDM2-3D-img224"
]

# Define target groups.
# (Only sex groups are used here.)
sex_groups = ["Male", "Female"]
organs = ["liver", "heart", "kidney", "spleen"]

# This dictionary will store, for each file, the data organized by group.
# Groups are defined as "Male liver", "Female liver", etc.
data_by_file = {}

# For the overall population split by organ, create a nested dictionary:
# overall_population_by_organ[organ][file] = list of records.
overall_population_by_organ = {organ: {} for organ in organs}
for organ in organs:
    for fname in files:
        overall_population_by_organ[organ][fname] = []

# Process each file.
for fname in files:
    # Full path for the file (adjust the path as needed).
    path_out=f"{external}/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/Generated_Data/recal/{fname}"
    with open(txt_data, 'r') as f:
        datalines = f.readlines()
    
    # Initialize a dictionary with an empty list for each group.
    data_by_group = {}
    for organ in organs:
        for sex in sex_groups:
            group_name = f"{sex} {organ}"
            data_by_group[group_name] = []
    
    # Process the file lines.
    # (Assumes each record spans 4 lines; adjust as needed.)
    for idx in range(0, len(datalines), 4):
        subiter = 0
        for organ in organs:
            try:
                line = datalines[idx + subiter]
            except IndexError:
                break  # Not enough lines left.
            subiter += 1
            
            # Split the line into fields.
            row = line.strip().split(', ')
            if len(row) < 9:
                continue
            age_str, group_str, sex_str, weight_str, height_str, organ_str, volume_str, suv_mean_str, suv_max_str = row
            
            # Process only if the organ in the row matches the current organ.
            if organ_str != organ:
                continue
            
            full_group = f"{sex_str} {organ_str}"
            
            # Convert numeric values.
            age = float(age_str)
            weight = float(weight_str)
            height = float(height_str)
            volume = float(volume_str)
            suv_mean = float(suv_mean_str)
            suv_max = float(suv_max_str)
            
            # Replace NaNs with 0.0.
            if math.isnan(suv_mean):
                suv_mean = 0.0
            if math.isnan(suv_max):
                suv_max = 0.0
            if math.isnan(volume):
                volume = 0.0          
            
            # Skip the record if volume is less than 50.
            if volume < 50:
                continue
            
            # (No age filtering here.)
            
            # Build the record.
            record = {
                "Age": age,
                "Weight": weight,
                "Height": height,
                "Volume": volume,
                "SUV_mean": suv_mean,
                "SUV_max": suv_max,
                "Organ": organ
            }
            
            # Append the record to the corresponding group.
            data_by_group[full_group].append(record)
            # Also add the record to the overall population for this organ.
            overall_population_by_organ[organ][fname].append(record)
    
    data_by_file[fname] = data_by_group

# -------------------------------------------------------------------
# Helper function to compute stats for a list of records.
def compute_stats(records):
    if not records:
        return 0, None, None, None, None, None, None
    subjects = len(records)
    ages = [r["Age"] for r in records]
    weights = [r["Weight"] for r in records]
    heights = [r["Height"] for r in records]
    volumes = [r["Volume"] for r in records]
    suv_means = [r["SUV_mean"] for r in records]
    suv_maxes = [r["SUV_max"] for r in records]
    
    avg_age, std_age = compute_average_std(ages)
    avg_weight, std_weight = compute_average_std(weights)
    avg_height, std_height = compute_average_std(heights)
    avg_volume, std_volume = compute_average_std(volumes)
    avg_suv_mean, std_suv_mean = compute_average_std(suv_means)
    avg_suv_max, std_suv_max = compute_average_std(suv_maxes)
    
    return subjects, (avg_age, std_age), (avg_weight, std_weight), (avg_height, std_height), (avg_volume, std_volume), (avg_suv_mean, std_suv_mean), (avg_suv_max, std_suv_max)

# -------------------------------------------------------------------
# 1. Group-wise Summary, ANOVA, and Statistical Testing
# Define the groups in a fixed order (we'll use all organs except the last one for demonstration).
groups = []
for organ in organs[:-1]:
    for sex in sex_groups:
        groups.append(f"{sex} {organ}")

for group in groups:
    table_data = []
    # Dictionary to store raw data per variable per file for statistical testing.
    test_data = {
        "Volume": {},
        "SUV_mean": {},
        "SUV_max": {}
    }
    
    for fname in files:
        records = data_by_file[fname].get(group, [])
        subjects, age_stat, weight_stat, height_stat, volume_stat, suv_mean_stat, suv_max_stat = compute_stats(records)
        
        if subjects == 0:
            row = [fname, "No data", "No data", "No data", "No data", "No data", "No data", "No data"]
        else:
            row = [
                fname,
                subjects,
                f"{age_stat[0]:.2f}±{age_stat[1]:.2f}",
                f"{weight_stat[0]:.2f}±{weight_stat[1]:.2f}",
                f"{height_stat[0]:.2f}±{height_stat[1]:.2f}",
                f"{volume_stat[0]:.2f}±{volume_stat[1]:.2f}",
                f"{suv_mean_stat[0]:.4f}±{suv_mean_stat[1]:.2f}",
                f"{suv_max_stat[0]:.4f}±{suv_max_stat[1]:.2f}"
            ]
        table_data.append(row)
        
        # Collect raw values for statistical tests.
        if records:
            test_data["Volume"][fname] = [r["Volume"] for r in records]
            test_data["SUV_mean"][fname] = [r["SUV_mean"] for r in records]
            test_data["SUV_max"][fname] = [r["SUV_max"] for r in records]
    
    headers = ["File", "Subjects", "Age (avg±std)", "Weight (avg±std)",
               "Height (avg±std)", "Volume (avg±std)", "SUV_mean (avg±std)", "SUV_max (avg±std)"]
    
    print(f"\nGroup: {group}")
    if tabulate:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        print(headers)
        for row in table_data:
            print(row)
    
    # -------------------------------
    # ANOVA Test for the group (for each variable)
    print("\nANOVA tests for group:", group)
    for var in ["Volume", "SUV_mean", "SUV_max"]:
        if len(test_data[var]) < 2:
            print(f"Not enough data for ANOVA on {var}.")
            continue
        samples = list(test_data[var].values())
        anova_result = f_oneway(*samples)
        print(f"{var} ANOVA: F = {anova_result.statistic:.2f}, p = {anova_result.pvalue:.4f}")
    
    # -------------------------------
    # T-Test comparisons against baseline for the group.
    baseline = "evaluation_log_Autopet.txt"
    print("\nT-Test comparisons (against evaluation_log_Autopet.txt) for group:", group)
    for var in ["Volume", "SUV_mean", "SUV_max"]:
        if baseline not in test_data[var]:
            print(f"Baseline file {baseline} has no data for {var}.")
            continue
        
        baseline_sample = test_data[var][baseline]
        baseline_mean = np.mean(baseline_sample)
        for fname in files:
            if fname == baseline:
                continue
            if fname in test_data[var]:
                sample = test_data[var][fname]
                ttest_result = ttest_ind(baseline_sample, sample, equal_var=False)
                sample_mean = np.mean(sample)
                if baseline_mean != 0:
                    perc_diff = ((sample_mean - baseline_mean) / baseline_mean) * 100
                else:
                    perc_diff = float('inf')
                print(f"{var} - {baseline} vs {fname}: t = {ttest_result.statistic:.2f}, p = {ttest_result.pvalue:.4f}, mean diff = {perc_diff:+.2f}%")
            else:
                print(f"{var} - No data in {fname} to compare with baseline.")

# -------------------------------------------------------------------
# 2. Entire-Population Summary, ANOVA, and Statistical Testing Split by Organ
for organ in organs[:-1]:
    print(f"\n=== Entire Population Summary for Organ: {organ} ===")
    overall_table_data = []
    overall_test_data = {
        "Volume": {},
        "SUV_mean": {},
        "SUV_max": {}
    }
    
    for fname in files:
        records = overall_population_by_organ[organ][fname]
        subjects, age_stat, weight_stat, height_stat, volume_stat, suv_mean_stat, suv_max_stat = compute_stats(records)
        
        if subjects == 0:
            row = [fname, "No data", "No data", "No data", "No data", "No data", "No data", "No data"]
        else:
            row = [
                fname,
                subjects,
                f"{age_stat[0]:.2f}±{age_stat[1]:.2f}",
                f"{weight_stat[0]:.2f}±{weight_stat[1]:.2f}",
                f"{height_stat[0]:.2f}±{height_stat[1]:.2f}",
                f"{volume_stat[0]:.2f}±{volume_stat[1]:.2f}",
                f"{suv_mean_stat[0]:.4f}±{suv_mean_stat[1]:.2f}",
                f"{suv_max_stat[0]:.4f}±{suv_max_stat[1]:.2f}"
            ]
        overall_table_data.append(row)
        
        if records:
            overall_test_data["Volume"][fname] = [r["Volume"] for r in records]
            overall_test_data["SUV_mean"][fname] = [r["SUV_mean"] for r in records]
            overall_test_data["SUV_max"][fname] = [r["SUV_max"] for r in records]
    
    headers = ["File", "Subjects", "Age (avg±std)", "Weight (avg±std)",
               "Height (avg±std)", "Volume (avg±std)", "SUV_mean (avg±std)", "SUV_max (avg±std)"]
    
    if tabulate:
        print(tabulate(overall_table_data, headers=headers, tablefmt="grid"))
    else:
        print(headers)
        for row in overall_table_data:
            print(row)
    
    # -------------------------------
    # ANOVA Test for overall population (organ-specific)
    print(f"\nANOVA tests for Entire Population (Organ: {organ})")
    for var in ["Volume", "SUV_mean", "SUV_max"]:
        if len(overall_test_data[var]) < 2:
            print(f"Not enough data for ANOVA on {var} for organ {organ}.")
            continue
        samples = list(overall_test_data[var].values())
        anova_result = f_oneway(*samples)
        print(f"{var} ANOVA: F = {anova_result.statistic:.2f}, p = {anova_result.pvalue:.4f}")
    
    # -------------------------------
    # T-Test comparisons for overall population of the organ.
    baseline = "evaluation_log_Autopet.txt"
    print(f"\nT-Test comparisons (against evaluation_log_Autopet.txt) for Organ: {organ}")
    for var in ["Volume", "SUV_mean", "SUV_max"]:
        if baseline not in overall_test_data[var]:
            print(f"Baseline file {baseline} has no data for {var} in organ {organ}.")
            continue
        
        baseline_sample = overall_test_data[var][baseline]
        baseline_mean = np.mean(baseline_sample)
        for fname in files:
            if fname == baseline:
                continue
            if fname in overall_test_data[var]:
                sample = overall_test_data[var][fname]
                ttest_result = ttest_ind(baseline_sample, sample, equal_var=False)
                sample_mean = np.mean(sample)
                if baseline_mean != 0:
                    perc_diff = ((sample_mean - baseline_mean) / baseline_mean) * 100
                else:
                    perc_diff = float('inf')
                print(f"{var} - {baseline} vs {fname}: t = {ttest_result.statistic:.2f}, p = {ttest_result.pvalue:.4f}, mean diff = {perc_diff:+.2f}%")
            else:
                print(f"{var} - No data in {fname} for organ {organ} to compare with baseline.")

# -------------------------------------------------------------------
# 3. Draw Violin Plots with Custom Colors (Male = Blue, Female = Red)
# For each organ, we will create one figure showing the data from each file.
# 3. Draw Violin Plots with Males First and Females Later for Each File
vars_to_plot = ["Volume", "SUV_mean", "SUV_max"]
gap = 0.0  # gap between male and female groups

for organ in organs[:-1]:
    for var in vars_to_plot:
        male_data = []
        male_labels = []
        female_data = []
        female_labels = []
        
        # Loop through files and collect data separately.
        for fname in files:
            male_records = data_by_file[fname].get(f"Male {organ}", [])
            female_records = data_by_file[fname].get(f"Female {organ}", [])
            
            # Skip if dataset is all zeros.
            if male_records:
                values = [r[var] for r in male_records]
                if not np.all(np.array(values) == 0.0):
                    male_data.append(values)
                    male_labels.append(fname)
            if female_records:
                values = [r[var] for r in female_records]
                if not np.all(np.array(values) == 0.0):
                    female_data.append(values)
                    female_labels.append(fname)
        
        # Skip if no data available for either group.
        if not male_data and not female_data:
            continue
        
        # Assign x positions: male positions 1..N, then female positions with a gap.
        n_male = len(male_data)
        n_female = len(female_data)
        male_positions = np.arange(1, n_male + 1)
        female_positions = np.arange(n_male + gap + 1, n_male + gap + n_female + 1)
        
        # Prepare combined lists for plotting.
        all_data = male_data + female_data
        all_positions = np.concatenate([male_positions, female_positions])
        all_labels = male_labels + female_labels
        
        plt.figure(figsize=(6,3))
        vp = plt.violinplot(all_data, positions=all_positions, showmeans=True, widths=0.8)
        
        # Color male violins blue, female violins red.
        for i, body in enumerate(vp['bodies']):
            if i < n_male:
                body.set_facecolor('blue')
                body.set_edgecolor('black')
                body.set_alpha(0.7)
            else:
                body.set_facecolor('red')
                body.set_edgecolor('black')
                body.set_alpha(0.7)
        
        # Draw a vertical dashed line to separate male and female groups.
        #if n_male > 0 and n_female > 0:
            #sep = n_male + gap/2
            #plt.axvline(sep, color='gray', linestyle='--', linewidth=1)
        
       #plt.xticks(all_positions, all_labels, rotation=45, ha='right')
        plt.title(f"Violin Plot of {var} for Organ: {organ}\n(Male = Blue, Female = Red)")
        plt.ylabel(var)
        #plt.xlabel("File")
        plt.tight_layout()
        plt.show()
