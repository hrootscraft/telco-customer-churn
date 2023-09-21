# refer jupyter notebook for code with context
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
import math
import pickle
import io

df = pd.read_excel("Telco_customer.xlsx")
target = "Churn Label"

df1 = df.copy()
# Dropping features
del df1["Country"]
del df1["State"]
del df1["CustomerID"]
del df1["Count"]
del df1["City"]
for col in [
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Score",
    "CLTV",
    "Churn Reason",
    "Zip Code",
]:
    del df1[col]

df1["Total Charges"] = (
    df1["Total Charges"].apply(str).str.strip()
)  # remove any leading and trailing whitespaces from each string in the 'Total Charges'
df1["Total Charges"] = pd.to_numeric(df1["Total Charges"])  # new dtype is float64
df1.replace(r"^\s*$", np.nan, regex=True, inplace=True)


def imputeTotalCharges(df):
    for index, row in df.iterrows():
        if math.isnan(row["Total Charges"]):
            df.at[index, "Total Charges"] = (
                df.at[index, "Monthly Charges"] * df.at[index, "Tenure Months"]
            )
    return df


df1 = imputeTotalCharges(df1)

df2 = df1.copy()
# Label Encoding
codes = {"Male": 0, "Female": 1}
df2["Gender"] = df2["Gender"].map(codes)

codes = {"No": 0, "Yes": 1}
for col in [
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Paperless Billing",
]:
    df2[col] = df2[col].map(codes)

codes = {"No": 0, "Yes": 1, "No phone service": 2}
df2["Multiple Lines"] = df2["Multiple Lines"].map(codes)

codes = {"DSL": 0, "Fiber optic": 1, "No": 2}
df2["Internet Service"] = df2["Internet Service"].map(codes)

codes = {"Yes": 0, "No": 1, "No internet service": 2}
for col in [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]:
    df2[col] = df2[col].map(codes)

codes = {"Month-to-month": 0, "Two year": 1, "One year": 2}
df2["Contract"] = df2["Contract"].map(codes)

codes = {
    "Mailed check": 0,
    "Electronic check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3,
}
df2["Payment Method"] = df2["Payment Method"].map(codes)

x_ols_features = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
]
y_ols = df2["Churn Value"]


def get_stats():
    x_ols = df2[x_ols_features]
    results = sm.OLS(y_ols, x_ols.astype(float)).fit()
    results_summary = results.summary()
    results_as_html = results_summary.tables[1].as_html()
    # result_sum = pd.read_html(results_as_html, header=0, index_col=0)[0]
    result_sum = pd.read_html(io.StringIO(results_as_html), header=0, index_col=0)[0]
    p_val = result_sum["P>|t|"]
    if max(p_val) <= 0.05:
        return x_ols_features, p_val, False
    m = "Column to be removed is " + str(p_val.idxmax())
    x_ols_features.remove(p_val.idxmax())
    return x_ols_features, p_val, True


con = True
while con:
    x_ols_features, p_val, con = get_stats()

df3 = df1[x_ols_features]
cat_cols = [cname for cname in df3.columns if df3[cname].dtype == "object"]

ohe = OneHotEncoder(sparse_output=False)
train_X_encoded = pd.DataFrame(ohe.fit_transform(df3[cat_cols]))
train_X_encoded.columns = ohe.get_feature_names_out(cat_cols)

df3 = df3.drop(cat_cols, axis=1).copy()
df4 = pd.concat([df3, train_X_encoded], axis=1)
df4[target] = y_ols

with open("ohe.pkl", "wb") as f:
    pickle.dump(ohe, f)

df4.to_csv("processed.csv")
