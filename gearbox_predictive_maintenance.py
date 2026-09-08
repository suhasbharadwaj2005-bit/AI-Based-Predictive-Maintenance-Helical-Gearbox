import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATASET_PATH = "data"

csv_files = glob.glob(os.path.join(DATASET_PATH, "*.csv"))

print("CSV files found:", len(csv_files))

if not csv_files:
    print("PHM 2009 Gearbox Dataset not found.")
else:
    data = []

    for file in csv_files:
        df = pd.read_csv(file)
        data.append(df)

    dataset = pd.concat(data, ignore_index=True)

    print("Dataset shape:", dataset.shape)
    print(dataset.head())
