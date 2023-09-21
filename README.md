# Telco Customer Churn
<sup>Supervised Machine Learning Project : An End-To-End Pipeline</sup>
### INTRODUCTION

**Did you know that attracting a new customer costs five times as much as keeping an existing one?**

- The telecommunications business has an annual churn rate of 15% - 25% in this highly competitive market.
- Corporations and businesses can forecast which customers are likely to leave ahead of time and focus on customer retention efforts.
- And as a result,
  - preserve their market position,
  - grow and thrive,
  - lower the cost of initiation,
  - larger the profit.

### DATA OVERVIEW
- Each row represents a customer, each column contains customer’s attributes described on the column Metadata.
- The data set includes information about :
  - Customers who left within the last month – the column is called Churn.
  - Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies.
  - Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges.
  - Demographic info about customers – gender, age range, and if they have partners and dependents.

### EDA REPORT
- 26.5 % Of Customers Switched To Another Firm.
  
![image](https://github.com/rbb-99/telco-customer-churn/assets/84140842/967c1774-da2a-47f6-a470-351727e68a07)

- Both Genders Behaved In Similar Fashion When It Comes To Migrating To Another Service Provider/Firm.

![image](https://github.com/rbb-99/telco-customer-churn/assets/84140842/8e352f19-4a2b-4bf7-9145-fbeca5a303bb)

- About 75% Of Customer With Month-To-Month Contract Opted To Move Out As Compared To 13% Of Customers With One Year Contract And 3% With Two Year Contract.

![image](https://github.com/rbb-99/telco-customer-churn/assets/84140842/3a2f1ab9-d0ed-4133-b79f-0b0e7d33216a)

- Major Customers Who Moved Out Had An Electronic Check As Payment Method On File Customers Who Opted For Credit-Card Automatic Transfer Or Bank Automatic Transfer And Mailed Check As Payment Method Were Less Likely To Move Out.

![image](https://github.com/rbb-99/telco-customer-churn/assets/84140842/cf8dc72e-6e21-4259-a155-a79427ab0a40)

- Customers With Higher Monthly Charges Are Also More Likely To Churn.

![image](https://github.com/rbb-99/telco-customer-churn/assets/84140842/3d8eb1d6-ff38-479c-8a5e-6b0bce1f8566)

- Most Of The Senior Citizens Churn.
- Customers With Paperless Billing Are Most Likely To Churn.

### DATA PRE-PROCESSING AND CLEANING
- Standard scalar to scale numerical columns down to the same range.
- Splitting the data into train and test sets.
- Manually categorizing the data in 0,1 form.
- One hot encoding the total charges column.
- Label encoding.
- Dropping the redundant columns such as country, state, count, latitude, longitude.

### ML MODEL EVALUATIONS AND PREDICTING
Now that the data is processed and cleaned, let’s start predicting the churn status.
- Random Forest Classifier Gives Best Prediction On Raw Unscaled Data With F1 Score Of 79% .
- Knn Classifier Gives A 78% F1 Score With Scaled Data.
- Random Forest Classifier Gives 77% F1 Score With Balanced, Scaled Data.
- Neural Networks Give The Best Accuracy Score Of 86% .

### COMPARATIVE EVALUATIONS OF ALL ML MODELS USED
Logistic regression, SVM Classifier, Random Forest, KNN, XGBoost Classifier, LightGBM Classifier tuned with best/recommended parameters using cross-validation.

### LOSS CURVES FOR TRAINING AND VALIDATION METRICS

### DEPLOYMENT USING STREAMLIT

### CONCLUSION
- The best way to avoid customer churn is to identify customers who are at risk of churning and working to improve their satisfaction.
- The most important features that helped this models are ”Tenure” which had the biggest effect and then “TechSupport” and “TotalCharges”
- Based on the results, Random Forests and Neural Network models predict the probability of “high risk” customers very effectively.
- ROC AUC was used as the evaluation metric
  - _suitable to classification problems_
  - _robust to imbalance of the target classes compared to accuracy_
- The confusion matrix  was used to check if for avoidance of both type I error and type II errors.

**References:**
- [Kaggle Dataset by Blastchar](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
