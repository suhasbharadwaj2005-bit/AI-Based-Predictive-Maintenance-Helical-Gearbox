import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


DATASET_PATH = "PHM_2009_Gearbox_Dataset"


def preprocess_signal(signal):
    signal = np.asarray(signal, dtype=float)
    signal = signal[~np.isnan(signal)]
    signal = signal - np.mean(signal)
    return signal


def extract_features(signal):
    signal = preprocess_signal(signal)

    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    peak_to_peak = np.ptp(signal)
    variance = np.var(signal)
    std = np.std(signal)
    kurt = kurtosis(signal)
    skewness = skew(signal)
    crest_factor = peak / rms if rms != 0 else 0

    return [
        rms,
        peak,
        peak_to_peak,
        variance,
        std,
        kurt,
        skewness,
        crest_factor
    ]


csv_files = glob.glob(
    os.path.join(DATASET_PATH, "**", "*.csv"),
    recursive=True
)

print("CSV files found:", len(csv_files))


feature_data = []

for file in csv_files:

    df = pd.read_csv(file, header=None)

    if df.shape[1] < 3:
        continue

    input_signal = df.iloc[:, 0].values
    output_signal = df.iloc[:, 1].values

    input_features = extract_features(input_signal)
    output_features = extract_features(output_signal)

    features = input_features + output_features

    feature_data.append(features)


feature_columns = [
    "input_rms",
    "input_peak",
    "input_peak_to_peak",
    "input_variance",
    "input_std",
    "input_kurtosis",
    "input_skewness",
    "input_crest_factor",
    "output_rms",
    "output_peak",
    "output_peak_to_peak",
    "output_variance",
    "output_std",
    "output_kurtosis",
    "output_skewness",
    "output_crest_factor"
]


features_df = pd.DataFrame(
    feature_data,
    columns=feature_columns
)

print("Feature dataset shape:", features_df.shape)
print(features_df.head())


# Add the verified PHM condition labels here
# after matching each recording with its labeled condition.

labels = pd.Series(dtype="object")


if len(labels) == len(features_df):

    X = features_df
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    decision_tree = DecisionTreeClassifier(
        random_state=42
    )

    random_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    decision_tree.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)

    dt_predictions = decision_tree.predict(X_test)
    rf_predictions = random_forest.predict(X_test)

    print("\nDecision Tree Accuracy:")
    print(accuracy_score(y_test, dt_predictions))

    print("\nRandom Forest Accuracy:")
    print(accuracy_score(y_test, rf_predictions))

    print("\nRandom Forest Confusion Matrix:")
    print(confusion_matrix(y_test, rf_predictions))

    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf_predictions))

else:
    print("\nFeature extraction completed.")
    print("Add the verified PHM condition labels before model training.")
