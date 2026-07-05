# 📡 Telco Customer Churn Predictor

An interactive web application built with Python, Streamlit, and Machine Learning to predict the likelihood of a telecom customer churning (leaving the service). 

## 🌟 Overview
Customer churn is a critical metric for subscription-based businesses. This app uses a machine learning model to analyze customer demographics, account information, and subscribed services to calculate a churn probability score and flag high-risk customers.

### Model Performance
The underlying model is a **Random Forest Classifier** trained using **SMOTE** (Synthetic Minority Over-sampling Technique) to effectively handle class imbalance. 
- **ROC AUC Score:** `0.8232` (82.3%)

## 📊 Key Insights from EDA
Our Exploratory Data Analysis (`EDA_Churn.ipynb`) uncovered several major drivers of customer churn:
- **Contract & Tenure:** Customers on `Month-to-month` contracts churn at an extremely high rate (42.7%). The vast majority of churn occurs within the first 12 months (median tenure of churners is just 10 months).
- **Service Type:** `Fiber optic` internet users have a high churn rate (41.9%), whereas DSL users are much more stable. Lack of add-ons (Tech Support, Online Security) also drastically increases churn probability.
- **Financial Friction:** The `Electronic check` payment method sees a massive 45.3% churn rate, suggesting potential friction or dissatisfaction.
- **Demographics:** Gender has no impact on churn, but customers *without* dependents or partners are twice as likely to leave.

**High-Risk Customer Profile:**
> A new customer (0-12 months tenure), single/without dependents, on a month-to-month contract, using Fiber Optic internet without tech support or online security, and paying via Electronic Check.

## 🛠️ Tech Stack
- **Frontend UI:** Streamlit
- **Machine Learning:** Scikit-Learn, Imbalanced-Learn
- **Data Manipulation:** Pandas, NumPy
- **Model Serialization:** Joblib

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aakash25-pixel/Churn_predictor.git
   cd Churn_predictor
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   The app will automatically open in your web browser at `http://localhost:8501`.

## 🧠 Retraining the Model
If you want to retrain the model from scratch on new data:
1. Ensure the dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) is downloaded.
2. Run the training script:
   ```bash
   python train_model.py
   ```
This will train the pipeline, output the latest F1 and ROC AUC scores, and generate a new `model.pkl` file which the app will automatically use!
