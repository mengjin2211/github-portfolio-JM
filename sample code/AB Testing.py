import random

def assign_group(user_id):
    return 'A' if random.random() < 0.5 else 'B'

users = [f"user_{i}" for i in range(1000)]
assignments = {user: assign_group(user) for user in users}
import numpy as np

 
np.random.seed(42)
data = {
    "group": [],
    "error_rate": [],
    "completion_rate": [],
    "time_to_completion": []
}

for user, group in assignments.items():
    data["group"].append(group)
    if group == "A":  # Current interface
        data["error_rate"].append(np.random.choice([0, 1], p=[0.7, 0.3]))
        data["completion_rate"].append(np.random.choice([0, 1], p=[0.8, 0.2]))
        data["time_to_completion"].append(np.random.normal(15, 5))
    else:  # Redesigned interface
        data["error_rate"].append(np.random.choice([0, 1], p=[0.85, 0.15]))
        data["completion_rate"].append(np.random.choice([0, 1], p=[0.9, 0.1]))
        data["time_to_completion"].append(np.random.normal(10, 3))

import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

df = pd.DataFrame(data)

from scipy.stats import chi2_contingency

# Error Rate Analysis
error_counts = df.groupby("group")["error_rate"].mean()
chi2, p_error, dof_error, expected_error = chi2_contingency(pd.crosstab(df["group"], df["error_rate"]))

# Completion Rate Analysis
completion_counts = df.groupby("group")["completion_rate"].mean()
chi2, p_completion, dof_completion, expected_completion = chi2_contingency(pd.crosstab(df["group"], df["completion_rate"]))

Time_counts = df.groupby("group")["time_to_completion"].mean()
# Separate the data into two groups
group1 = df[df["group"] == "A"]["time_to_completion"] 
group2 = df[df["group"] == "B"]["time_to_completion"] 

# Perform the independent t-test
t_stat, p_value = ttest_ind(group1, group2)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

print("Error Rate (Group A vs B):", error_counts)
print("p-value for Error Rate:", p_error)
print("Completion Rate (Group A vs B):", completion_counts)
print("p-value for Completion Rate:", p_completion)
print("Completion Time (Group A vs B):", Time_counts)
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")