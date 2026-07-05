# 📡 Telco Customer Churn Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://churnpredictor-jxpgsusdz7jl4k68mz3xtr.streamlit.app/)

An interactive web application built with Python, Streamlit, and Machine Learning to predict the likelihood of a telecom customer churning (leaving the service). 

## 🌟 The Problem Statement
Customer churn is a critical metric for subscription-based businesses. Retaining an existing customer is significantly cheaper than acquiring a new one. This project aims to build a predictive model that analyzes customer demographics, account information, and subscribed services to flag high-risk customers before they cancel their subscriptions, enabling targeted retention strategies.

---

## 📊 Exploratory Data Analysis (EDA) Insights
Our comprehensive EDA (`EDA_Churn.ipynb`) uncovered several major drivers of customer churn:

- **Contract & Tenure:** Customers on `Month-to-month` contracts churn at an extremely high rate (42.7%). The vast majority of churn occurs within the first 12 months (median tenure of churners is just 10 months).
- **Service Type:** `Fiber optic` internet users have a disproportionately high churn rate (41.9%), whereas DSL users are much more stable. Lack of add-ons (Tech Support, Online Security) also drastically increases churn probability.
- **Financial Friction:** The `Electronic check` payment method sees a massive 45.3% churn rate, suggesting potential friction, hidden fees, or dissatisfaction with that payment gateway.
- **Demographics:** Gender has almost no impact on churn, but customers *without* dependents or partners are twice as likely to leave compared to those with families.

> [!IMPORTANT]
> **High-Risk Customer Profile:** A new customer (0-12 months tenure), single/without dependents, on a month-to-month contract, using Fiber Optic internet without tech support or online security, and paying via Electronic Check.

---

## 🧠 Machine Learning Methodology

### 1. Data Preprocessing
- **Missing Values:** Addressed missing values in the `TotalCharges` column using median imputation.
- **Encoding:** Categorical variables were converted using `OneHotEncoder` to ensure the model interprets them correctly without assuming artificial ordinal relationships.
- **Scaling:** Numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) were standardized using `StandardScaler` to ensure features with larger scales do not dominate the algorithm.

### 2. Handling Class Imbalance: Why SMOTE?
In this dataset, only **~26.5%** of customers actually churned, while **~73.5%** were retained. This is a classic example of an **imbalanced dataset**. 

If we train a model directly on this raw data, the model will naturally become biased toward predicting "No Churn" simply because it is the vast majority. A naive model could guess "No Churn" every time and achieve 73.5% accuracy by completely ignoring the churning customers—which defeats the entire purpose of the business problem!

To solve this, we used **SMOTE (Synthetic Minority Over-sampling Technique)**. 
Instead of simply duplicating existing churn records (which leads to severe overfitting), SMOTE creates *synthetic, realistic* examples of churners by interpolating between existing minority class instances. This provides the model with a perfectly balanced 50/50 dataset during training, forcing it to actually learn the true underlying patterns and behaviors of a churning customer.

### 3. Model Selection & Evaluation
We utilized a **Random Forest Classifier** because it handles non-linear relationships well, is robust to outliers, and provides clear feature importances.

Instead of relying on basic Accuracy (which is deceptive on imbalanced data), we evaluate our model using metrics better suited for this specific problem:
- **ROC AUC Score:** `0.8232` (82.3%) — This metric proves our model has an excellent ability to distinguish and rank customers who will churn versus those who will stay.
- **F1-Score:** By utilizing SMOTE, our model successfully maintains a strong balance between Precision (not falsely alarming the business with fake churners) and Recall (successfully capturing the actual churners).

---

## 🛠️ Tech Stack
- **Frontend UI:** Streamlit
- **Machine Learning:** Scikit-Learn, Imbalanced-Learn (SMOTE)
- **Data Manipulation & Visualization:** Pandas, NumPy, Matplotlib, Seaborn
- **Model Serialization:** Joblib

---

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

---

## 🔄 Retraining the Model
If you want to retrain the model from scratch on new data:
1. Ensure the dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) is downloaded.
2. Run the training script:
   ```bash
   python train_model.py
   ```
This will train the Imblearn Pipeline (preprocessing -> SMOTE -> Random Forest), output the latest evaluation metrics, and generate a new `model.pkl` file which the app will automatically detect and load!
