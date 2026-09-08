# AI-Based Predictive Maintenance of an Industrial Helical Gearbox Using Machine Learning

## Mechanical Concepts Used

This project combines mechanical engineering principles with machine learning for gearbox condition monitoring.

Key mechanical concepts include:

* Helical gears and gear meshing
* Gearbox components and operation
* Gearbox vibration and condition monitoring
* Gearbox fault and failure mechanisms
* Preventive and predictive maintenance
* Vibration-based fault diagnosis

## Dataset

The project uses the **PHM Society 2009 Gearbox Dataset**, a publicly available experimental dataset developed for gearbox fault detection and condition monitoring.

The dataset contains synchronous measurements collected from an experimental gearbox test setup.

## Data Used

The PHM 2009 dataset includes:

* Input-side accelerometer signal
* Output-side accelerometer signal
* Tachometer signal

The experimental setup includes different operating conditions and gear configurations, including **spiral-cut (helical) gears**.

## Feature Extraction

Vibration signals are processed to extract statistical condition-monitoring features:

* RMS
* Peak
* Peak-to-Peak
* Variance
* Standard Deviation
* Kurtosis
* Skewness
* Crest Factor

These features represent different characteristics of the gearbox vibration signal and are used as inputs for machine-learning models.

## Machine Learning

The extracted vibration features are used to develop a classification-based predictive-maintenance model.

Models used:

* Decision Tree Classifier
* Random Forest Classifier

The models are trained and evaluated to classify the available gearbox operating or fault conditions.

## Project Objective

The objective is to demonstrate how vibration-based condition-monitoring data can be combined with machine learning to support **early gearbox fault detection and predictive maintenance**.

## Dataset Source

**PHM Society 2009 Data Challenge – Gearbox Dataset**

Official source:
https://phmsociety.org/public-data-sets/

The original dataset files are not included in this repository because of their size.
