# Diabetes Prediction using Machine Learning

## 📌 Project Overview
This project predicts whether a person is diabetic or not using machine learning techniques. The model is trained on the Pima Indians Diabetes dataset and uses Logistic Regression for classification.

## 📊 Features
- Data cleaning (handling invalid zero values)
- Exploratory Data Analysis (EDA)
- Data visualization (heatmaps, distributions)
- Feature scaling using StandardScaler
- Model training using Logistic Regression
- Model evaluation (accuracy, classification report)
- Sample prediction system

## 🛠️ Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## 📈 Model Performance
- Accuracy: ~70–80% (may vary slightly depending on split)

## 📁 Project Structure

```
Diabetes-Prediction/
├── data/
│   └── diabetes.csv            # Raw dataset
├── notebook/
│   └── eda.ipynb               # EDA + model building notebook
├── src/
│   └── train.py                # Model training script (optional)
├── venv/                       # Virtual environment (ignored in Git)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Ignored files
└── README.md                   # Project documentation
``` 

## 🧪 Sample Prediction
The model takes input features like glucose level, BMI, age, etc., and predicts whether a person is diabetic or not.

## 🎯 Future Improvements
- Try advanced models like Random Forest, SVM, XGBoost
- Hyperparameter tuning
- Build a web app using Streamlit

## 👩‍💻 Author
Shanvi Agarwal