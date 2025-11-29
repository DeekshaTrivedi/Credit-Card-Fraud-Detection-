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
