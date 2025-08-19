# Iris Classification API

## Problem Description
This project implements a FastAPI application for classifying Iris flower species (setosa, versicolor, virginica) using the Iris dataset (https://www.kaggle.com/datasets/uciml/iris). The task is a multiclass classification problem using four numerical features (sepal length, sepal width, petal length, petal width) plus two engineered features (sepal area, petal area).

## Model Choice Justification
- **Model**: RandomForestClassifier (scikit-learn, 100 estimators).
- **Why?**: Robust for classification, handles non-linear relationships, and achieves high accuracy (~95-98% on Iris). Preferred over LogisticRegression for better performance on this dataset.
- **Data Cleaning**: Removed duplicates (few in Iris dataset) and outliers using IQR on numerical features. No missing values found.
- **Feature Engineering**: Added `sepal_area` (SepalLengthCm * SepalWidthCm) and `petal_area` (PetalLengthCm * PetalWidthCm) to capture geometric relationships. Scaled all features using StandardScaler.
- **Assumptions/Limitations**:
  - Assumes clean input measurements (no negative or extreme values).
  - Iris dataset is small but well-balanced; model may overfit slightly without further tuning.
  - Accuracy is high, but confidence scores depend on model probabilities.

## How to Run the Application
1. Install dependencies: `pip install -r requirements.txt`
2. Run the notebook (`iris_classification.ipynb`) to clean data, engineer features, train, and save `model.pkl` and `scaler.pkl`.
3. Start the API: `uvicorn main:app --reload`
4. Test endpoints: http://localhost:8000/docs

## API Usage Examples
- **Health Check (GET /)**:
  - Request: `curl -X GET "http://localhost:8000/"`
  - Response: `{"status": "healthy", "message": "Iris Classification API is running"}`

- **Predict (POST /predict)**:
  - Example 1: Setosa-like
    - Request: `curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"SepalLengthCm": 5.1, "SepalWidthCm": 3.5, "PetalLengthCm": 1.4, "PetalWidthCm": 0.2}'`
    - Response: `{"prediction": "Iris-setosa", "confidence": 0.98}`
  - Example 2: Virginica-like
    - Request: `curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"SepalLengthCm": 6.7, "SepalWidthCm": 3.0, "PetalLengthCm": 5.2, "PetalWidthCm": 2.3}'`
    - Response: `{"prediction": "Iris-virginica", "confidence": 0.95}`

- **Model Info (GET /model-info)**:
  - Request: `curl -X GET "http://localhost:8000/model-info"`
  - Response: `{"model_type": "RandomForestClassifier", "problem_type": "classification", "features": ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "sepal_area", "petal_area"]}`
