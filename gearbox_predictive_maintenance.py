import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import kurtosis, skew

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


np.random.seed(42)

SAMPLES_PER_SIGNAL = 2000
NUMBER_OF_SIGNALS_PER_CLASS = 100
SAMPLING_FREQUENCY = 1000

CONDITIONS = [
    "Normal",
    "Bearing Fault",
    "Gear Wear",
    "Severe Fault"
]


def generate_vibration_signal(condition, signal_length=2000):

    t = np.arange(signal_length) / SAMPLING_FREQUENCY

    base_frequency = 50

    base_signal = (
        0.5 * np.sin(2 * np.pi * base_frequency * t)
    )

    noise = np.random.normal(
        0,
        0.15,
        signal_length
    )

    signal = base_signal + noise

    if condition == "Normal":

        signal += np.random.normal(
            0,
            0.05,
            signal_length
        )

    elif condition == "Bearing Fault":

        signal = (
            1.3 * signal
            + np.random.normal(
                0,
                0.20,
                signal_length
            )
        )

        fault_interval = 100

        for i in range(
            fault_interval,
            signal_length,
            fault_interval
        ):

            end = min(i + 10, signal_length)
            impulse_length = end - i

            signal[i:end] += (
                1.5
                * np.exp(
                    -np.linspace(
                        0,
                        4,
                        impulse_length
                    )
                )
            )

    elif condition == "Gear Wear":

        signal = (
            1.7 * signal
            + np.random.normal(
                0,
                0.25,
                signal_length
            )
        )

        gear_frequency = 150

        signal += (
            0.5
            * np.sin(
                2
                * np.pi
                * gear_frequency
                * t
            )
        )

    elif condition == "Severe Fault":

        signal = (
            2.5 * signal
            + np.random.normal(
                0,
                0.40,
                signal_length
            )
        )

        fault_interval = 70

        for i in range(
            fault_interval,
            signal_length,
            fault_interval
        ):

            end = min(i + 20, signal_length)
            impulse_length = end - i

            signal[i:end] += (
                2.5
                * np.exp(
                    -np.linspace(
                        0,
                        3,
                        impulse_length
                    )
                )
            )

    return signal


def preprocess_signal(signal):

    signal = np.asarray(
        signal,
        dtype=float
    )

    signal = signal[
        ~np.isnan(signal)
    ]

    signal = (
        signal
        - np.mean(signal)
    )

    return signal


def extract_features(signal):

    signal = preprocess_signal(signal)

    rms = np.sqrt(
        np.mean(
            signal ** 2
        )
    )

    peak = np.max(
        np.abs(signal)
    )

    peak_to_peak = np.ptp(
        signal
    )

    variance = np.var(
        signal
    )

    std = np.std(
        signal
    )

    kurt = kurtosis(
        signal
    )

    skewness = skew(
        signal
    )

    if rms != 0:
        crest_factor = peak / rms
    else:
        crest_factor = 0

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


feature_data = []
labels = []


for condition in CONDITIONS:

    for _ in range(
        NUMBER_OF_SIGNALS_PER_CLASS
    ):

        input_signal = generate_vibration_signal(
            condition,
            SAMPLES_PER_SIGNAL
        )

        output_signal = generate_vibration_signal(
            condition,
            SAMPLES_PER_SIGNAL
        )

        input_features = extract_features(
            input_signal
        )

        output_features = extract_features(
            output_signal
        )

        features = (
            input_features
            + output_features
        )

        feature_data.append(
            features
        )

        labels.append(
            condition
        )


features_df = pd.DataFrame(
    feature_data,
    columns=feature_columns
)

labels = pd.Series(
    labels,
    name="Condition"
)

dataset = features_df.copy()

dataset["Condition"] = labels


print("Dataset shape:", dataset.shape)

print("\nFirst five rows:")
print(dataset.head())

print("\nClass distribution:")
print(labels.value_counts())


X = features_df
y = labels


print("\nNumber of input features:", X.shape[1])
print("Number of samples:", X.shape[0])
print("\nTarget classes:")
print(y.unique())


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


decision_tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=8
)

decision_tree.fit(
    X_train,
    y_train
)


random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10
)

random_forest.fit(
    X_train,
    y_train
)


dt_predictions = decision_tree.predict(
    X_test
)

rf_predictions = random_forest.predict(
    X_test
)


dt_accuracy = accuracy_score(
    y_test,
    dt_predictions
)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)


print(
    f"\nDecision Tree Accuracy: "
    f"{dt_accuracy * 100:.2f}%"
)

print(
    f"Random Forest Accuracy: "
    f"{rf_accuracy * 100:.2f}%"
)


rf_cm = confusion_matrix(
    y_test,
    rf_predictions,
    labels=CONDITIONS
)


print("\nRandom Forest Confusion Matrix:")

print(
    pd.DataFrame(
        rf_cm,
        index=CONDITIONS,
        columns=CONDITIONS
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        rf_predictions
    )
)


feature_importance = pd.DataFrame({

    "Feature": feature_columns,

    "Importance":
        random_forest.feature_importances_

})


feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print("\nFeature Importance:")

print(
    feature_importance
)


new_input_signal = generate_vibration_signal(
    "Gear Wear",
    SAMPLES_PER_SIGNAL
)

new_output_signal = generate_vibration_signal(
    "Gear Wear",
    SAMPLES_PER_SIGNAL
)


new_input_features = extract_features(
    new_input_signal
)

new_output_features = extract_features(
    new_output_signal
)


new_features = (
    new_input_features
    + new_output_features
)


new_sample = pd.DataFrame(
    [new_features],
    columns=feature_columns
)


print("\nNew gearbox vibration features:")

print(
    new_sample
)


new_prediction = random_forest.predict(
    new_sample
)

prediction_probabilities = (
    random_forest.predict_proba(
        new_sample
    )[0]
)


predicted_condition = new_prediction[0]


print("\nPredicted Gearbox Condition:")
print(predicted_condition)


print("\nPrediction probabilities:")

for condition, probability in zip(
    random_forest.classes_,
    prediction_probabilities
):

    print(
        f"{condition:20s} : "
        f"{probability * 100:.2f}%"
    )


print("\nMaintenance Decision:")

if predicted_condition == "Normal":

    print(
        "Gearbox condition appears normal."
    )

elif predicted_condition == "Bearing Fault":

    print(
        "Inspect bearings and lubrication."
    )

elif predicted_condition == "Gear Wear":

    print(
        "Inspect gear teeth, lubrication and alignment."
    )

elif predicted_condition == "Severe Fault":

    print(
        "Immediate inspection recommended."
    )


normal_signal = generate_vibration_signal(
    "Normal",
    SAMPLES_PER_SIGNAL
)

fault_signal = generate_vibration_signal(
    "Severe Fault",
    SAMPLES_PER_SIGNAL
)


time = (
    np.arange(
        SAMPLES_PER_SIGNAL
    )
    / SAMPLING_FREQUENCY
)


plt.figure(
    figsize=(10, 5)
)

plt.plot(
    time,
    normal_signal
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Vibration Amplitude"
)

plt.title(
    "Normal Gearbox Vibration Signal"
)

plt.grid(
    True
)

plt.tight_layout()

plt.show()


plt.figure(
    figsize=(10, 5)
)

plt.plot(
    time,
    fault_signal
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Vibration Amplitude"
)

plt.title(
    "Severe Fault Gearbox Vibration Signal"
)

plt.grid(
    True
)

plt.tight_layout()

plt.show()


plt.figure(
    figsize=(10, 7)
)

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()


model_names = [
    "Decision Tree",
    "Random Forest"
]

model_accuracies = [
    dt_accuracy * 100,
    rf_accuracy * 100
]


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    model_names,
    model_accuracies
)

plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Model Accuracy Comparison"
)

plt.ylim(
    0,
    100
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.show()


print("\n======================================================")
print("PROJECT COMPLETED")
print("======================================================")

print(
    "\nProject: AI-Based Predictive Maintenance "
    "of an Industrial Gearbox"
)

print("\nMachine Learning Models:")
print("1. Decision Tree")
print("2. Random Forest")

print(
    "\nNumber of Features:",
    len(feature_columns)
)

print("\nFinal predicted gearbox condition:")
print(predicted_condition)

print("\n======================================================")
