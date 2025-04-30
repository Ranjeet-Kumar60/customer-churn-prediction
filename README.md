# 📦 Customer Churn Prediction

This project predicts customer churn for a telecom company using a machine learning pipeline. A real-time prediction web app is also built with Streamlit and deployed on Render.

---

## 🔍 Project Overview

- **Objective**: Identify customers likely to churn based on their usage patterns and service information.
- **Impact**: Enables early intervention and retention efforts, helping reduce customer loss and protect revenue.
- **Tools**: Python, Pandas, Scikit-learn, Optuna, SMOTE, Streamlit

---

## ⚙️ Highlights

- 🔹 Cleaned & preprocessed the Telco dataset.
- 🔹 Handled imbalanced data using SMOTE.
- 🔹 Selected Top 10 features using Random Forest.
- 🔹 Tuned Logistic Regression using Optuna for hyperparameters.
- 🔹 Trained final model and saved it as `.pkl`.
- 🔹 Created a Streamlit web app that:
  - Takes 20 raw user inputs.
  - Preprocesses them using saved pipeline.
  - Selects only top 10 features for prediction.
- 🔹 Deployed on Render for real-time churn prediction.

---

## 🗂 Project Structure

```
├── app.py                       # Streamlit app
├── final_model.pkl              # Trained logistic regression model
├── preprocessor.pkl             # Preprocessing pipeline (scaler + encoder)
├── top_10_features.pkl          # Selected top features
├── requirements.txt             # List of dependencies
├── README.md                    # Project documentation
├── data/
│   ├── train.csv                # Training data
│   └── test.csv                 # Testing data
├── *.ipynb                      # Training, feature selection, and tuning notebooks
```

---

## 🛠 Setup Instructions

1. **Clone this repo**
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
streamlit run app.py
```

---

## 🚀 Access the Application

👉 Open your browser and visit:  
**https://customer-churn-prediction-app.onrender.com**  
(*replace with your actual link after deployment*)

---

## 🛠 How to Use the App

1. Fill in the customer details using dropdowns or text fields.
2. Click **Predict** to get the churn result.
3. The app uses preprocessing and top 10 selected features to make accurate predictions.

---

## 📷 Screenshots

### ✅ Streamlit App Interface
![App Screenshot](https://user-images.githubusercontent.com/your_screenshot_path_here)

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── data/
│   ├── train.csv
│   └── test.csv
├── app.py
├── final_model.pkl
├── preprocessor.pkl
├── top_10_features.pkl
├── requirements.txt
├── README.md
└── *.ipynb files (EDA, model training, etc.)
```

---

## ✅ Features

- Cleaned and preprocessed Telco churn dataset
- Feature selection with Random Forest (Top 10 features)
- Logistic Regression + SMOTE + Optuna hyperparameter tuning
- Threshold tuning to maximize **Recall**
- Final model saved using **joblib**
- Streamlit-based UI for user-friendly prediction
- Deployed on **Render**

---

## 🔮 Future Improvements

- Add SHAP/LIME for model explainability  
- CRM integration for real-time data usage  
- Schedule retraining with new data  
- Interactive threshold control for business teams  
- Visual dashboard with churn metrics  

---

## 🙌 Acknowledgements

- [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)  
- Streamlit for UI  
- Optuna for hyperparameter tuning  
- Scikit-learn for ML modeling  

## 📈 Model Performance

- **Model**: Logistic Regression (SMOTE + Optuna tuned)
- **Recall**: ~82%
- **AUC**: ~84%
- **Features Used**: Top 10 selected from original 20

---

## 🌐 Deployment

- Streamlit app deployed on Render
- User inputs processed via trained pipeline
- Prediction returned in real time

---

## 📊 Dataset

- Source: Telco Customer Churn Dataset
- Converted to `train.csv` and `test.csv` after cleaning and splitting

---

## 🚀 Future Improvements

- ✅ **Add Model Explainability**: Use SHAP or LIME to explain predictions and increase trust in the model.
- ✅ **Feedback Loop**: Collect user feedback from app predictions to retrain and improve the model over time.
- ✅ **Support Multiple Models**: Allow switching between models (e.g., Random Forest, LightGBM) via dropdown.
- ✅ **Dashboard Integration**: Build a dashboard to monitor prediction volume, churn rate, and performance metrics.
- ✅ **CRM Integration**: Connect the app to real customer databases or CRM systems for business use.
- ✅ **Auto Retraining**: Schedule periodic retraining with updated data to maintain model accuracy.
- ✅ **Customizable Threshold**: Allow users to adjust threshold interactively to suit business needs.

---

## 📜 License
- This project is free to use under the MIT License.

---

## 🌟 Show Your Support
- ⭐ If you like this project, give it a star on GitHub!
- 💬 Have feedback or suggestions? Feel free to contribute!

---

## 📬 Author

Developed by **Ranjeet Kumar**  
🔗 GitHub: [Ranjeet-Kumar60](https://github.com/Ranjeet-Kumar60)

--- 

