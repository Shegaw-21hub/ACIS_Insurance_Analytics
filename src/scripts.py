# 1.Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.getcwd())  # Ensure it aligns with your project's location

# 2.Load Data

# Updated path with forward slashes
data_path = "d:/Project/ACIS_Insurance_Analytics/data/MachineLearningRating_v3.txt"
df = pd.read_csv(data_path, sep="|", low_memory=False)

# Read the file with pandas
df = pd.read_csv(data_path, sep="|", low_memory=False)

# Save processed data
df.to_csv("d:/Project/ACIS_Insurance_Analytics/data/clean.csv", index=False)


# 4.Data Cleaning and Feature Engineering
df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce', format='%Y-%m')
df = df[df['TotalPremium'] > 0].copy()
df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']




