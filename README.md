# CodeAlpha-Credit-Scoring-Model
Machine Learning based Credit Scoring Prediction System using Random Forest and Streamlit.
# 💳  Scoring Prediction System

## Project Overview
This project uses Machine Learning to predict a customer's credit risk based on financial and personal information.

The goal is to classify customers into Good Credit or Bad Credit categories using a trained Machine Learning model.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learnCredit
- Streamlit
- Joblib

## Dataset
- Dataset: German Credit Dataset
- Source: UCI Machine Learning Repository
- Records: 1,000
- Features: 20

## Machine Learning Models
The following models were explored:
- Logistic Regression
- Decision Tree
- Random Forest

A tuned Random Forest model was saved for prediction.

## Project Structure

```text
CodeAlpha_Credit_Scoring_Model/
│
├── app.py
├── task1.ipynb
├── README.md
│
├── dataset/
│   └── credit_data.csv
│
└── models/
    └── random_forest.pkl
```

## How to Run the Project

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

### 2. Run the application

```bash
python -m streamlit run app.py
```

### 3. Open in browser

```text
http://localhost:8501
```

## Application Features
- Enter customer financial information.
- Predict the credit class.
- View prediction probabilities.
- Use an interactive Streamlit interface.

## Future Improvements
- Improve model performance.
- Add more model evaluation visualizations.
- Improve the user interface.
- Deploy the application online.

## Author
Machine Learning Project – CodeAlpha Internship
## Model Evaluation Results

The tuned Random Forest model was evaluated on a test dataset containing 200 records.

| Metric | Bad Credit | Good Credit |
|---|---:|---:|
| Precision | 0.53 | 0.84 |
| Recall | 0.68 | 0.74 |
| F1-score | 0.59 | 0.79 |

**Overall Accuracy: 72.0%**

### Interpretation
- The model correctly classified 72% of the test records.
- It achieved 84% precision for Good Credit predictions.
- It achieved 68% recall for Bad Credit cases.

These results show that the model is a useful learning prototype, but it needs further improvement before any real credit decisions. 
## Confusion Matrix

The confusion matrix visualizes the performance of the Random Forest classification model.

![Random Forest Confusion Matrix](confusion_matrix.png)

