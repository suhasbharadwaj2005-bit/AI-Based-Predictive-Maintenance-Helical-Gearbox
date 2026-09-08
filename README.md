# AI-Based Predictive Maintenance of an Industrial Helical Gearbox Using Machine Learning

## Project Overview

Gearboxes are critical mechanical components used to transmit power and motion in industrial machinery. Continuous operation can lead to faults such as gear tooth damage, wear, misalignment, and other abnormal conditions.

This project develops a **machine-learning-based predictive maintenance prototype** for an industrial helical gearbox using vibration condition-monitoring data.

The project combines **mechanical engineering, vibration analysis, signal processing, and machine learning** to classify gearbox conditions and demonstrate how data-driven methods can support maintenance decisions.

## Objective

The main objectives of this project are to:

* Analyze gearbox vibration condition-monitoring data.
* Extract meaningful statistical features from vibration signals.
* Develop machine-learning models for gearbox condition classification.
* Compare Decision Tree and Random Forest classification models.
* Demonstrate the application of machine learning in predictive maintenance.

## Mechanical Concepts

The project is based on the following mechanical engineering concepts:

* Helical gears and gear meshing
* Industrial gearbox operation
* Gearbox components
* Gearbox vibration
* Gearbox fault mechanisms
* Condition monitoring
* Preventive maintenance
* Predictive maintenance
* Vibration-based fault diagnosis

## Dataset

The project uses the **PHM Society 2009 Gearbox Dataset**, a publicly available experimental dataset developed for gearbox fault detection and condition monitoring.

The experimental setup contains:

* Input-side accelerometer measurements
* Output-side accelerometer measurements
* Tachometer measurements

The dataset includes different operating conditions and gear configurations, including **spiral-cut (helical) gears**.

## Project Workflow

```text
PHM 2009 Gearbox Dataset
          ↓
Vibration Signal Acquisition
          ↓
Signal Preprocessing
          ↓
Feature Extraction
          ↓
Feature Dataset
          ↓
Train-Test Split
          ↓
Decision Tree / Random Forest
          ↓
Model Evaluation
          ↓
Gearbox Condition Classification
```

## Signal Processing and Feature Extraction

The raw vibration signals are preprocessed before feature extraction.

The following statistical features are extracted:

| Feature            | Purpose                                           |
| ------------------ | ------------------------------------------------- |
| RMS                | Represents the overall vibration magnitude        |
| Peak               | Represents the maximum vibration amplitude        |
| Peak-to-Peak       | Represents the total signal amplitude range       |
| Variance           | Represents signal variability                     |
| Standard Deviation | Measures deviation from the mean                  |
| Kurtosis           | Indicates impulsive characteristics of the signal |
| Skewness           | Represents signal asymmetry                       |
| Crest Factor       | Relates peak amplitude to RMS value               |

These features convert the raw vibration signals into numerical indicators suitable for machine-learning models.

## Machine Learning Models

### Decision Tree

A Decision Tree Classifier is used as a simple and interpretable baseline model for gearbox condition classification.

### Random Forest

A Random Forest Classifier combines multiple decision trees to improve classification performance and robustness.

The models are trained using the extracted vibration features and evaluated on unseen test data.

## Model Evaluation

The classification models are evaluated using:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-Score
* Classification Report

The evaluation helps determine how effectively the models distinguish between the available gearbox conditions.

## Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **SciPy**
* **Scikit-learn**
* **Matplotlib**

## Project Structure

```text
AI-Based-Predictive-Maintenance-Helical-Gearbox/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── README.md
│
└── gearbox_predictive_maintenance.py
```

## Applications

The developed prototype demonstrates the potential use of machine learning for:

* Gearbox condition monitoring
* Early fault detection
* Vibration-based diagnosis
* Maintenance decision support
* Predictive maintenance systems

## Limitations

* The project is based on publicly available experimental data.
* The model classifies the conditions represented in the dataset.
* The system does not replace physical inspection or maintenance expertise.
* Real industrial deployment would require continuous sensor acquisition, validation, and testing under actual operating conditions.

## Future Improvements

Future development could include:

* Real-time vibration sensor integration
* Continuous condition monitoring
* Additional machine-learning models
* Real industrial gearbox data
* Real-time predictive-maintenance dashboard
* Remaining Useful Life (RUL) estimation

## Dataset Reference

**PHM Society 2009 Data Challenge – Gearbox Dataset**

Official source:

https://phmsociety.org/public-data-sets/
