# ANFIS Tide Level Prediction App

## Overview
This Streamlit application provides a user-friendly interface for predicting tide levels using a pre-trained ANFIS (Adaptive Neuro-Fuzzy Inference System) model.

## Features
- Single input prediction
- Batch prediction from CSV files
- Visualization of actual vs predicted values
- Error analysis and results download

## Setup Instructions
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place your pre-trained `anfis_model.pkl` in the same directory
4. Run the app: `streamlit run app.py`

## Required CSV Format
For batch prediction, your CSV should have columns:
- `x1 (2 jam sebelumnya)`: Input 2 hours before
- `x2 (1 jam sebelumnya)`: Input 1 hour before
- `y (pasang surut saat ini)`: Actual tide level (for error calculation)

## Model
Trained using Recursive Least Squares Estimation (RLSE) with a Fuzzy Neural Network approach.

## How to run
1. Make virtual environment
```
py -m venv venv
```

2. Activate virtual environment
```
./venv/Scripts/Activate
```
3. Install dependencies: 
```
pip install -r requirements.txt
```
4. Make sure your pre-trained `anfis_model.pkl` in the same directory.

5. Run the app:
```
streamlit run app.py
```

6. Run with logging: 
```
streamlit run --logger.level=debug app.py
```