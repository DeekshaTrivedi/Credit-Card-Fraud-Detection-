# Credit Card Fraud Analysis

A comprehensive exploration of large-scale credit-card transaction data to understand fraud behavior, engineer meaningful features, and prepare the dataset for machine-learning–based fraud detection.  
This project covers data cleaning, exploratory data analysis (EDA), visualization, correlation studies, fraud-pattern identification, and construction of derived features based on financial logic.

---

## Project Overview

Financial fraud is rare but dangerous, often hidden behind irregular transaction patterns and broken balance flows.  
This project works through a full analytical workflow to uncover these patterns and build a foundation for fraud-detection modeling.

### Goals
- Explore the dataset and understand variable relationships  
- Identify fraud-prone transaction types  
- Study behavioral differences between fraudulent and legitimate transactions  
- Visualize transaction distributions using log transformations  
- Build derived features exposing irregular money flow  
- Prepare data for future machine-learning models

---

## Dataset Information

The dataset contains 6.36 million credit-card transactions with the following key columns:

- step – Hour of the simulation  
- type – Transaction category (TRANSFER, CASH_OUT, PAYMENT, etc.)  
- amount – Transaction value  
- nameOrig – Sender account ID  
- oldbalanceOrg / newbalanceOrig – Sender balance before/after  
- nameDest – Receiver account ID  
- oldbalanceDest / newbalanceDest – Receiver balance before/after  
- isFraud – True fraud label  
- isFlaggedFraud – System-flagged fraud (very rare)

Fraud appears only in TRANSFER and CASH_OUT transactions.

---

## Data Cleaning

- Checked for missing values (none present)  
- Converted data types where needed  
- Removed non-useful columns (optional: step)  
- Ensured balance-flow consistency  
- Filtered transaction types for targeted fraud analysis  

---

## Exploratory Data Analysis

### Skewed Transaction Amounts
Transaction values were highly skewed, so log transformations were applied:

```python
np.log1p(df["amount"])
```

This produced distributions that were easier to interpret and visualize.

## Fraud Distribution

- isFraud = 1: 8,213 cases
- isFraud = 0: 6,354,407 cases

Fraud rate is approximately 0.12%, showing strong class imbalance.

## Transaction Type Analysis

Fraud occurs exclusively in:

- TRANSFER

- CASH_OUT

##Correlation Analysis

A correlation matrix was generated to study relationships among numerical features:
```python
df[cols].corr()

```
Key observations:
- No strong linear correlation exists with fraud
- Financially paired features show expected high correlations
- Fraud behavior is non-linear, making engineered features important

## Feature Engineering

To capture financial inconsistencies, the following features were created:
```python
df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]
```
These help expose:
- Broken balance logic
- Unexpected negative or zero balances
- Suspicious transaction patterns

##Fraud Pattern Insights

- Important findings from the analysis:
- Fraud sender accounts rarely repeat
- Fraud receiver accounts often appear hundreds of times
- Many fraudulent transactions leave senders with newbalanceOrig == 0
- A large group of suspicious cases show sender balance draining unexpectedly
- Engineered features successfully highlight hidden irregularities

An example mask used to detect suspicious transfers:
```python
(df["oldbalanceOrg"] > 0) &
(df["newbalanceOrig"] == 0) &
(df["type"].isin(["TRANSFER", "CASH_OUT"]))
```

##Model Training

A full machine-learning pipeline was created using:

- ColumnTransformer
- StandardScaler
- OneHotEncoder
- LogisticRegression / RandomForest / XGBoost (depending on experiment)

The final trained pipeline was saved as:
```python
joblib.dump(pipeline, "fraud_detection_model.pkl")
```
This .pkl file contains:

- preprocessing steps
- encoding
- scaling
- the trained classifier
This allows seamless use inside the Streamlit app.

##Streamlit Fraud Detection App
A simple web interface was built using Streamlit to allow users to input transaction details and receive a fraud prediction.
##
##Running the Streamlit App
###Install dependencies:
```bash
pip install -r requirements.txt
```
###Run the app:
```bash
streamlit run fraud_detection.py
```
###Required File
Ensure the model file exists:
```bash
fraud_detection_model.pkl
```
Streamlit loads this pipeline automatically.

###The app will open in your browser.
Input transaction details, and the model will predict whether it is fraudulent.


###Clone the Repository
```bash
git clone <your-repo-link>
```
###Install Dependencies
```bash
pip install -r requirements.txt
```

###Launch Jupyter
```bash
jupyter notebook
```

Run all cells in the notebook sequentially.
