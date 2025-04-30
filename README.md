# 📦 Customer Churn Prediction

This project predicts customer churn for a telecom company using a machine learning model. It also includes a Streamlit web app deployed on Render for real-time predictions.

---

## 🔍 Project Overview

- **Goal**: Find out which customers are likely to leave.
- **Why it matters**: Helps the company stop churn and save revenue.
- **Tools used**: Python, Pandas, Scikit-learn, SMOTE, Optuna, Streamlit

---

## ✅ Key Highlights

- Cleaned and prepared Telco churn dataset
- Used SMOTE to fix class imbalance
- Selected top 10 features using Random Forest
- Tuned Logistic Regression with Optuna
- Saved final model and preprocessor using `joblib`
- Built a Streamlit app that:
  - Takes 20 raw inputs from user
  - Preprocesses them using saved pipeline
  - Picks only top 10 features for prediction
- Deployed the app on Render

---

## 📁 Folder Structure

```
customer-churn-prediction/
├── app.py                  # Streamlit app
├── final_model.pkl         # Trained model
├── preprocessor.pkl        # Preprocessing pipeline
├── top_10_features.pkl     # Selected features
├── requirements.txt        # Python dependencies
├── README.md               # Project info
├── data/
│   ├── train.csv           # Training data
│   └── test.csv            # Testing data
├── *.ipynb                 # Notebooks for training and analysis
```

---

## 🛠 How to Set Up

1. **Clone the repo**
```bash
git clone https://github.com/Ranjeet-Kumar60/customer-churn-prediction.git
cd customer-churn-prediction
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Run the app locally**
```bash
streamlit run app.py
```

---

## 🚀 Web App Access

Visit:  
**https://customer-churn-prediction-odfr.onrender.com**  

---

## 💡 How the App Works

1. Fill in customer details on the web form.
2. Click on **Predict**.
3. It uses the saved pipeline to clean and process inputs.
4. Then it uses the top 10 features to predict if a customer will churn.

---

## 📷 Screenshot

### ✅ Streamlit App Interface
https://github.com/Ranjeet-Kumar60/customer-churn-prediction/blob/main/streamlit%20app%20screenshot/Screenshot%202025-05-01%20005454.png
https://github.com/Ranjeet-Kumar60/customer-churn-prediction/blob/main/streamlit%20app%20screenshot/Screenshot%202025-05-01%20005519.png

---

## 📊 Model Performance

- **Model**: Logistic Regression
- **Tuning**: SMOTE + Optuna
- **Recall**: ~82%
- **AUC**: ~84%
- **Final Features Used**: Top 10 from original 20

---

## 🌐 Deployment Info

- App is built using Streamlit
- Deployed on **Render**
- Predictions are made in real-time from user input

---

## 🔮 Future Improvements

- Add SHAP or LIME to explain predictions
- Add feature to collect user feedback and retrain the model
- Allow user to choose between multiple models (e.g., RF, LightGBM)
- Add live dashboard to track performance
- Link with CRM system for real-time usage
- Schedule auto-retraining with new data
- Let user control prediction threshold in UI

---

## 🙌 Acknowledgements

- [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)  
- Streamlit  
- Scikit-learn  
- Optuna

---

## 📜 License

This project is under the **MIT License**. You can use and modify it freely.

---

## 💬 Contact

Built by **Ranjeet Kumar**  
🔗 GitHub: [Ranjeet-Kumar60](https://github.com/Ranjeet-Kumar60)

If you liked this project, feel free to ⭐ it or share your feedback!
