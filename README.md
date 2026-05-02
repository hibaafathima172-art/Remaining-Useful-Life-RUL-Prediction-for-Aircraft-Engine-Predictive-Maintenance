# Remaining-Useful-Life-RUL-Prediction-for-Aircraft-Engine-Predictive-Maintenance

This project predicts the **Remaining Useful Life (RUL)** of aircraft engines using machine learning. It is based on the NASA C-MAPSS FD001 dataset and includes both a machine learning training pipeline and an interactive Streamlit dashboard.

The system predicts how many operating cycles an aircraft engine may continue to function before failure or maintenance is required. Based on the predicted RUL, the dashboard classifies the engine condition as **Healthy**, **Warning**, or **Critical** and provides a prediction summary, key drivers, interpretation, and maintenance recommendation.

---

## Project Description

Aircraft engines generate sensor data during operation. These sensor readings change gradually as the engine degrades over time. By analysing these patterns, machine learning models can estimate the Remaining Useful Life of an engine.

This project uses historical run-to-failure data from aircraft engines to train regression models. The trained model is then deployed using Streamlit so that users can enter engine sensor values and receive a prediction.

The main goal of this project is to support **predictive maintenance** by identifying potential engine degradation before failure occurs.

---

## Objectives

The objectives of this project are:

- To load and preprocess the NASA C-MAPSS FD001 dataset
- To calculate Remaining Useful Life for each engine cycle
- To train machine learning models for RUL prediction
- To compare SVR, Random Forest, and XGBoost models
- To select the best-performing model based on evaluation metrics
- To save the trained model and preprocessing objects
- To create a user-friendly Streamlit dashboard
- To provide prediction summary, key drivers, interpretation, and recommendation

---

## Dataset Used

The project uses the **NASA C-MAPSS FD001 dataset**.

The dataset contains simulated aircraft engine degradation data. Each engine starts in a healthy condition and gradually degrades until failure.

![Streamlit Prediction Output](RUL1.png)
![Streamlit Prediction Output](RUL2.png)
![Streamlit Prediction Output](RUL3.png)
![Streamlit Prediction Output](RUL4.png)
![Streamlit Prediction Output](RUL5.png)

The dataset includes:

- Engine ID
- Cycle number
- 3 operational settings
- 21 sensor readings

The main files used are:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt

| File                                | Description                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------------ |
| `main.py`                           | Main Streamlit dashboard application                                           |
| `engine_rul_model_training.ipynb`   | Notebook containing data preprocessing, model training, evaluation, and saving |
| `rul_model.pkl`                     | Saved best model                                                               |
| `scaler.pkl`                        | Saved scaler used for input transformation                                     |
| `features.pkl`                      | Saved list of model feature names                                              |
| `default_profiles.pkl`              | Saved default engine health profiles                                           |
| `requirements.txt`                  | Python dependencies required to run the project                                |
| `README.md`                         | Project documentation                                                          |
| `screenshots/prediction_output.png` | Screenshot of the running Streamlit dashboard                                  |


