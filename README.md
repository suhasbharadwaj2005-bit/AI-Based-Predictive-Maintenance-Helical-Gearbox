# AI-Based Predictive Maintenance of an Industrial Helical Gearbox

## Project Overview

Predictive maintenance uses machine condition data to identify possible faults before they lead to major machine failure.

This project develops a machine learning-based prototype for **condition monitoring of an industrial helical gearbox using vibration signal analysis**.

The system generates simulated gearbox vibration signals for different operating conditions, extracts statistical features from the vibration signals, and uses machine learning models to classify the gearbox condition.

### Operating Conditions

* Normal
* Bearing Fault
* Gear Wear
* Severe Fault

> **Note:** The current prototype uses simulated vibration signals. It is intended to demonstrate the complete predictive-maintenance workflow and is not a real industrial diagnostic system.

---

## Objective

The main objectives of this project are:

* Generate simulated gearbox vibration signals.
* Preprocess the vibration signals.
* Extract statistical features from vibration signals.
* Train machine learning models for gearbox condition classification.
* Compare Decision Tree and Random Forest models.
* Predict the condition of a new gearbox vibration signal.
* Generate a basic maintenance recommendation based on the predicted condition.

---

## Project Workflow

```text
Simulated Gearbox Vibration Signals
                ↓
        Signal Preprocessing
                ↓
       Feature Extraction
                ↓
     16 Statistical Features
                ↓
        Train-Test Split
                ↓
      Machine Learning Models
        ↙               ↘
Decision Tree       Random Forest
        ↓               ↓
        Model Evaluation
                ↓
       New Vibration Signal
                ↓
       Condition Prediction
                ↓
      Maintenance Decision
```

---

## Technologies Used

* Python
* NumPy
* Pandas
* SciPy
* Scikit-learn
* Matplotlib

---

## Vibration Signal Processing

The simulated vibration signals contain a base vibration component and condition-specific changes.

Different fault conditions are represented by changes in vibration characteristics such as:

* Vibration amplitude
* Noise level
* Impulsive behavior
* Additional frequency components

The signals are then preprocessed by removing the mean value before feature extraction.

---

## Feature Extraction

Eight statistical features are extracted from each vibration signal.

### Input-Side Features

1. RMS
2. Peak
3. Peak-to-Peak
4. Variance
5. Standard Deviation
6. Kurtosis
7. Skewness
8. Crest Factor

### Output-Side Features

The same eight features are extracted from the output-side vibration signal.

Therefore:

**8 input-side features + 8 output-side features = 16 features**

### Complete Feature List

```text
input_rms
input_peak
input_peak_to_peak
input_variance
input_std
input_kurtosis
input_skewness
input_crest_factor

output_rms
output_peak
output_peak_to_peak
output_variance
output_std
output_kurtosis
output_skewness
output_crest_factor
```

---

## Machine Learning Models

Two classification algorithms were implemented:

### 1. Decision Tree

A Decision Tree classifies gearbox conditions by learning decision rules from the extracted vibration features.

### 2. Random Forest

Random Forest combines multiple decision trees to make the final classification.

The Random Forest model was also used to calculate feature importance and predict the condition of a new simulated gearbox vibration signal.

---

## Dataset

The prototype generated:

* **400 total vibration signals**
* **100 signals per condition**
* **4 gearbox conditions**
* **16 input features**

### Dataset Distribution

| Condition     | Number of Samples |
| ------------- | ----------------: |
| Normal        |               100 |
| Bearing Fault |               100 |
| Gear Wear     |               100 |
| Severe Fault  |               100 |
| **Total**     |           **400** |

The dataset was divided using an 80:20 train-test split.

| Dataset  | Samples |
| -------- | ------: |
| Training |     320 |
| Testing  |      80 |

---

## Obtained Results

### Model Accuracy

| Model         |    Accuracy |
| ------------- | ----------: |
| Decision Tree | **100.00%** |
| Random Forest | **100.00%** |

The Random Forest model correctly classified all **80 test samples** in the simulated test dataset.

### Random Forest Confusion Matrix

| Actual / Predicted | Normal | Bearing Fault | Gear Wear | Severe Fault |
| ------------------ | -----: | ------------: | --------: | -----------: |
| Normal             |     20 |             0 |         0 |            0 |
| Bearing Fault      |      0 |            20 |         0 |            0 |
| Gear Wear          |      0 |             0 |        20 |            0 |
| Severe Fault       |      0 |             0 |         0 |           20 |

### Classification Report

| Condition     | Precision | Recall | F1-Score |
| ------------- | --------: | -----: | -------: |
| Bearing Fault |      1.00 |   1.00 |     1.00 |
| Gear Wear     |      1.00 |   1.00 |     1.00 |
| Normal        |      1.00 |   1.00 |     1.00 |
| Severe Fault  |      1.00 |   1.00 |     1.00 |

---

## Feature Importance

The Random Forest model identified the following features as having relatively higher importance:

| Feature             | Importance |
| ------------------- | ---------: |
| input_variance      |   0.106575 |
| output_peak_to_peak |   0.096307 |
| output_std          |   0.093871 |
| output_rms          |   0.088210 |
| input_std           |   0.077054 |

This indicates that vibration magnitude-related statistical features contributed significantly to the classification performed on the simulated dataset.

---

## New Gearbox Prediction

A new simulated gearbox vibration signal was generated and processed using the same feature extraction procedure.

### Prediction

**Predicted Gearbox Condition: Gear Wear**

### Prediction Probability

```text
Bearing Fault : 0.00%
Gear Wear     : 100.00%
Normal        : 0.00%
Severe Fault  : 0.00%
```

### Maintenance Recommendation

```text
Inspect gear teeth, lubrication and alignment.
```

This demonstrates how the prototype can connect vibration-based condition classification with a basic maintenance recommendation.

---

## Results Visualization

The project generates the following plots:

### Normal Vibration Signal

![Normal Vibration](results/normal_vibration.png)

### Severe Fault Vibration Signal

![Severe Fault Vibration](results/severe_fault_vibration.png)

### Feature Importance

![Feature Importance](results/feature_importance.png)

### Model Accuracy Comparison

![Model Accuracy](results/model_accuracy_comparison.png)

---

## Project Structure

```text
AI-Based-Predictive-Maintenance-Helical-Gearbox/
│
├── gearbox_predictive_maintenance.py
├── README.md
├── requirements.txt
│
└── results/
    ├── normal_vibration.png
    ├── severe_fault_vibration.png
    ├── feature_importance.png
    └── model_accuracy_comparison.png
```

---

## How to Run

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Open the project folder

```bash
cd AI-Based-Predictive-Maintenance-Helical-Gearbox
```

### 3. Install the required Python libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Python program

```bash
python gearbox_predictive_maintenance.py
```

The program generates the simulated vibration data, extracts features, trains the machine learning models, evaluates their performance, predicts a new gearbox condition, and generates the result plots.

---

## Limitations

The current project is a prototype and has the following limitations:

* The vibration signals are simulated rather than collected from a physical gearbox.
* The fault characteristics are simplified.
* The model has been evaluated only on the simulated dataset.
* Real industrial vibration signals may contain more complex noise and operating conditions.
* The current system performs fault classification rather than remaining useful life prediction.

Therefore, the **100% accuracy obtained in this prototype should not be interpreted as real-world industrial diagnostic accuracy**.

---

## Future Improvements

Possible future developments include:

* Collecting real gearbox vibration data using accelerometers.
* Testing the model with experimental or industrial datasets.
* Adding real-time vibration acquisition.
* Implementing real-time gearbox condition monitoring.
* Adding more gearbox fault types.
* Comparing additional machine learning and deep learning algorithms.
* Developing a real-time monitoring dashboard.
* Implementing Remaining Useful Life (RUL) prediction.
* Integrating the system with an industrial IoT platform.

---

## Conclusion

This project demonstrates a complete prototype workflow for **machine-learning-based predictive maintenance of an industrial gearbox**.

Vibration signals are processed to extract statistical features, which are then used by Decision Tree and Random Forest models to classify gearbox operating conditions.

The prototype successfully demonstrated classification of four simulated gearbox conditions and prediction of a new simulated vibration signal as **Gear Wear**.

The project combines **mechanical engineering concepts such as gearbox condition monitoring and vibration analysis with Python and machine learning**, providing a foundation for future Industry 4.0 predictive-maintenance applications.
