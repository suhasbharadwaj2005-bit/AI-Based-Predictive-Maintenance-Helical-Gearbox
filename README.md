# AI-Based-Predictive-Maintenance-Helical-Gearbox
Machine Learning based predictive maintenance prototype for an industrial helical gearbox using vibration condition-monitoring data.

# Dataset

## PHM Society 2009 Gearbox Dataset

This project uses the **PHM Society 2009 Gearbox Dataset**, a publicly available experimental dataset for gearbox fault detection and condition monitoring.

The dataset contains vibration and tachometer measurements collected from an experimental gearbox setup.

### Data Used

The dataset includes:

* Input-side accelerometer signal
* Output-side accelerometer signal
* Tachometer signal

The experiments were performed under different gearbox operating conditions and gear configurations, including **spiral-cut (helical) gears**.

### Feature Extraction

The raw vibration signals are processed to extract condition-monitoring features such as:

* RMS
* Peak
* Peak-to-Peak
* Variance
* Standard Deviation
* Kurtosis
* Skewness
* Crest Factor

These features are used as inputs to the machine-learning models.

### Machine Learning

The extracted features are used to train classification models for identifying the available gearbox conditions.

The models used in this project are:

* Decision Tree
* Random Forest

### Dataset Source

**PHM Society 2009 Data Challenge – Gearbox Dataset**

Official source:
https://phmsociety.org/public-data-sets/

The original dataset files are not included in this repository because of their size.

---

**Done by Suhas M**
