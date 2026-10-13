import streamlit as st
import multiprocessing

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA
import lightgbm
import xgboost as xgb

from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc

import statsmodels.api as sm
from imblearn.over_sampling import SMOTE

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

import pickle

percent = 0
df = pd.read_csv("processed.csv")

X = df.drop("Churn Label", axis=1)
y = df["Churn Label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=20
)

X_train = X_train.reset_index(drop=True)
del X_train["Unnamed: 0"]
X_test = X_test.reset_index(drop=True)
del X_test["Unnamed: 0"]


def predict_new(model_name, model, X_train, y_train, X_test, y_test):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        senior_citizen = st.selectbox("Are they a senior citizen?", ["Yes", "No"])
    with col2:
        partner = st.selectbox("Do they have a partner?", ["Yes", "No"])
    with col3:
        dependent = st.selectbox("Do they have a dependent?", ["Yes", "No"])
    with col4:
        phone_services = st.selectbox(
            "Have they subscibed to phone services?", ["Yes", "No"]
        )
    with col5:
        internet_services = st.selectbox(
            "Have they subscibed to internet services?", ["DSL", "Fiber optic", "No"]
        )

    col11, col12, col13, col14, col15 = st.columns(5)
    with col11:
        online_security = st.selectbox(
            "Have they subscibed to online security?",
            ["Yes", "No", "No internet services"],
        )
    with col12:
        online_backup = st.selectbox(
            "Have they subscibed to online backup?",
            ["Yes", "No", "No internet services"],
        )
    with col13:
        device_protection = st.selectbox(
            "Have they subscibed to device protection?",
            ["Yes", "No", "No internet services"],
        )
    with col14:
        tech_support = st.selectbox(
            "Have they subscibed to tech support?",
            ["Yes", "No", "No internet services"],
        )
    with col15:
        contract = st.selectbox(
            "What type of contract do they have?",
            ["Month-to-month", "Two Year", "One Year"],
        )

    col21, col22 = st.columns(2)
    with col21:
        paperless_billing = st.selectbox(
            "Do they have paperless billing", ["Yes", "No"]
        )
    with col22:
        payment_method = st.selectbox(
            "What's their main payment method?",
            [
                "Mailed check",
                "Electronic check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    col23, col24, col25 = st.columns(3)
    with col23:
        months = st.number_input(
            "Enter number of months they were active", step=1, min_value=0
        )
    with col24:
        monthly_payment = st.number_input(
            "Enter their monthly payment", step=1, min_value=0
        )
    with col25:
        Total_payment = st.number_input(
            "Enter the total amount they paid", step=1, min_value=0
        )

    if st.button("Predict Now"):
        data = []
        data.append(
            [
                months,
                monthly_payment,
                Total_payment,
                senior_citizen,
                partner,
                dependent,
                phone_services,
                internet_services,
                online_security,
                online_backup,
                device_protection,
                tech_support,
                contract,
                paperless_billing,
                payment_method,
            ]
        )

        all_cols = [
            "Tenure Months",
            "Monthly Charges",
            "Total Charges",
            "Churn Label",
            "Senior Citizen_No",
            "Senior Citizen_Yes",
            "Partner_No",
            "Partner_Yes",
            "Dependents_No",
            "Dependents_Yes",
            "Phone Service_No",
            "Phone Service_Yes",
            "Internet Service_DSL",
            "Internet Service_Fiber optic",
            "Internet Service_No",
            "Online Security_No",
            "Online Security_No internet service",
            "Online Security_Yes",
            "Online Backup_No",
            "Online Backup_No internet service",
            "Online Backup_Yes",
            "Device Protection_No",
            "Device Protection_No internet service",
            "Device Protection_Yes",
            "Tech Support_No",
            "Tech Support_No internet service",
            "Tech Support_Yes",
            "Contract_Month-to-month",
            "Contract_One year",
            "Contract_Two year",
            "Paperless Billing_No",
            "Paperless Billing_Yes",
            "Payment Method_Bank transfer (automatic)",
            "Payment Method_Credit card (automatic)",
            "Payment Method_Electronic check",
            "Payment Method_Mailed check",
        ]

        df1 = pd.DataFrame(
            data,
            columns=[
                "Tenure Months",
                "Monthly Charges",
                "Total Charges",
                "Senior Citizen",
                "Partner",
                "Dependents",
                "Phone Service",
                "Internet Service",
                "Online Security",
                "Online Backup",
                "Device Protection",
                "Tech Support",
                "Contract",
                "Paperless Billing",
                "Payment Method",
            ],
        )

        with open("ohe.pkl", "rb") as pickle_file:
            ohe = pickle.load(pickle_file)
        cat_columns = [cname for cname in df1.columns if df1[cname].dtype == "object"]
        train_X_encoded = pd.DataFrame(ohe.fit_transform(df1[cat_columns]))
        train_X_encoded.columns = ohe.get_feature_names_out(cat_columns)
        df1 = df1.drop(cat_columns, axis=1).copy()
        df2 = pd.concat([df1, train_X_encoded], axis=1)

        for c in all_cols:
            if c not in df2.columns:
                df2[c] = 0

        df3 = df2.copy()
        df3 = df3.drop("Churn Label", axis=1).copy()
        df_col = list(df.columns)
        df_col.remove("Churn Label")
        df3 = df3.reindex(columns=df_col)
        df3 = df3.drop("Unnamed: 0", axis=1)
        print(df3.columns)
        pred = model.predict_proba(df3)
        print(pred)
        st.write("The probability of customer churning is ", pred[0][1])


def plot_loss_curves(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    epochs = range(len(history.history["loss"]))
    col1, col2 = st.columns(2)
    with col1:
        # Plot loss
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, loss, label="training_loss")
        plt.plot(epochs, val_loss, label="val_loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.legend()
        st.pyplot(plt)
    with col2:
        # Plot accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, accuracy, label="training_accuracy")
        plt.plot(epochs, val_accuracy, label="val_accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()
        st.pyplot(plt)


def run(X_train, y_train, X_test, y_test, d):
    model = st.selectbox(
        "Which models do you want to run?",
        [
            "",
            "Ensemble",
            "Logistic Regression",
            "SVM",
            "Random Forest",
            "KNN",
            "xgboost",
            "Lightboost",
            "Neural Network",
        ],
    )

    def evaluate(typ, model, X_train, y_train, x_test, y_test):
        if typ == "nn":
            pred = model.predict(X_test)
            pred = [1 if x > 0.5 else 0 for x in pred]
            pred = pd.Series(pred)
            pred = pred.astype(int)
        else:
            pred = model.predict(x_test)

        col1, col2 = st.columns(2)
        with col1:
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, pred)
            ax = plt.subplot()
            sns.heatmap(cm, annot=True, ax=ax, fmt="g")
            ax.set_xlabel("Predicted", fontsize=20)
            ax.xaxis.set_label_position("top")
            ax.xaxis.set_ticklabels(["Not Churned", "Churned"], fontsize=15)
            ax.xaxis.tick_top()
            ax.set_ylabel("True", fontsize=20)
            ax.yaxis.set_ticklabels(["Not Churned", "Churned"], fontsize=15)
            st.pyplot(plt)
        with col2:
            fpr, tpr, threshold = roc_curve(y_test, pred)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8, 6))
            plt.title("Receiver Operating Characteristic")
            plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
            plt.legend(loc="lower right")
            plt.plot([0, 1], [0, 1], "r--")
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel("True Positive Rate")
            plt.xlabel("False Positive Rate")
            st.pyplot(plt)
        return round(f1_score(y_test, pred, average="weighted"), 5)

    st.write("Running model ", model)

    if len(model) > 0:
        if model == "Logistic Regression":
            par = st.selectbox(
                "What model parameters do you want?",
                ("", "Recommended", "Custom"),
            )
            if len(par) > 0:
                if par == "Recommended":
                    c = 1.0
                    solver = "liblinear"
                    best_params = d["Logistic Regression"]
                else:
                    c = st.slider(
                        "What value of c do you want to use?",
                        min_value=0.001,
                        max_value=10.0,
                        step=0.1,
                    )
                    st.write(
                        "liblinear tries for l1,l2 and sage tries for ElasticNet, l1,l2,none"
                    )
                    solver = st.selectbox(
                        "Which solver do you want to use?", ["liblinear", "saga"]
                    )
                    best_params = {"C": c, "solver": solver}
                lr = LogisticRegression(**best_params, class_weight="balanced")
                lr.fit(X_train, y_train)
                lr_f1 = evaluate("lr", lr, X_train, y_train, X_test, y_test)
                st.write("Model trained with an F1 score of", lr_f1)
                predict_new("lr", lr, X_train, y_train, X_test, y_test)

        if model == "SVM":
            par = st.selectbox(
                "What model parameters do you want?",
                ("", "Recommended", "Custom"),
            )
            if len(par) > 0:
                if par == "Recommended":
                    best_params = d["SVM"]
                else:
                    c = st.slider(
                        "What value of c do you want to use?",
                        min_value=0.001,
                        max_value=10.0,
                        step=0.1,
                    )
                    gamma = st.slider(
                        "What value of gamma do you want to use?",
                        min_value=0.001,
                        max_value=1.0,
                        step=0.0001,
                    )
                    st.write(
                        "liblinear tries for l1,l2 and sage tries for ElasticNet, l1,l2,none"
                    )
                    kernel = st.selectbox(
                        "What kernel value do you want to use?",
                        ["rbf", "poly", "linear", "sigmoid"],
                    )
                    best_params = {"C": c, "gamma": gamma, "kernel": kernel}
                svm = SVC(**best_params)
                svm.fit(X_train, y_train)
                print("Model fit")
                svm_f1 = evaluate("svm", svm, X_train, y_train, X_test, y_test)
                st.write("Model trained with an F1 score of", svm_f1)
                print(svm_f1)
                predict_new("svm", svm, X_train, y_train, X_test, y_test)

        if model == "Random Forest":
            best_params = d["Random Forest"]
            rfc = RandomForestClassifier(**best_params)
            rfc.fit(X_train, y_train)
            rf_f1 = evaluate("rf", rfc, X_train, y_train, X_test, y_test)
            st.write("Model trained with an F1 score of", rf_f1)
            predict_new("rfc", rfc, X_train, y_train, X_test, y_test)

        if model == "KNN":
            par = st.selectbox(
                "What kind of model parameters do you want?",
                ("", "Recommended", "Custom"),
            )
            if len(par) > 0:
                if par == "Recommended":
                    n = 7
                else:
                    n = st.slider("What value of n?", min_value=1, max_value=20, step=1)
                knn = KNeighborsClassifier(n_neighbors=n)
                knn.fit(X_train, y_train)
                knn_f1 = evaluate("knn", knn, X_train, y_train, X_test, y_test)
                st.write("Model trained with an F1 score of", knn_f1)
                predict_new("knn", knn, X_train, y_train, X_test, y_test)

        if model == "xgboost":
            best_params = d["xgboost"]
            xgb_m = xgb.XGBClassifier(**best_params)
            xgb_m.fit(X_train, y_train)
            xgb_f1 = evaluate("xgb", xgb_m, X_train, y_train, X_test, y_test)
            st.write("Model trained with an F1 score of", xgb_f1)
            predict_new("xgb_m", xgb_m, X_train, y_train, X_test, y_test)

        if model == "Lightboost":
            best_params = d["Lightboost"]
            lgb = lightgbm.LGBMClassifier(**best_params)
            lgb.fit(X_train, y_train)
            lgb_f1 = evaluate("lgb", lgb, X_train, y_train, X_test, y_test)
            st.write("Model trained with an F1 score of", lgb_f1)
            predict_new("lgb", lgb, X_train, y_train, X_test, y_test)

        if model == "Ensemble":
            lr = LogisticRegression(**d["Logistic Regression"])
            svm = SVC(**d["SVM"])
            lgb = lightgbm.LGBMClassifier(**d["Lightboost"])
            rfc = RandomForestClassifier(**d["Random Forest"])
            xgb_m = xgb.XGBClassifier(**d["xgboost"])
            knn = KNeighborsClassifier(n_neighbors=6)
            estimators = [
                ("lr", lr),
                ("svm", svm),
                ("rfc", rfc),
                ("xgb_m", xgb_m),
                ("knn", knn),
            ]
            ensemble = VotingClassifier(estimators, voting="hard")
            ensemble.fit(X_train, y_train)
            pred = ensemble.predict(X_test)
            ensemble_f1 = evaluate(
                "ensemble", ensemble, X_train, y_train, X_test, y_test
            )
            st.write("Model trained with an F1 score of", ensemble_f1)
            predict_new("ensemble", ensemble_f1, X_train, y_train, X_test, y_test)

        if model == "Neural Network":
            input_layer = Input(shape=(X_train.shape[1],))
            par = st.selectbox(
                "What kind of model parameters do you want?",
                ("", "Recommended", "Custom"),
            )
            if len(par) > 0:
                if par == "Recommended":
                    dense_layer_1 = Dense(100, activation="sigmoid")(input_layer)
                    dense_layer_2 = Dense(100, activation="sigmoid")(dense_layer_1)
                    output_layer = Dense(1, activation="sigmoid")(dense_layer_2)
                    model = Model(inputs=input_layer, outputs=output_layer)
                    epochs = 10
                else:
                    layers = st.number_input(
                        "Enter the number of hidden layers",
                        step=1,
                        min_value=1,
                        max_value=10,
                    )
                    epochs = st.number_input(
                        "Enter the number of epochs", step=1, min_value=2, max_value=50
                    )
                    layers = int(layers)
                    units = [0] * layers
                    activation = [""] * layers

                    for i in range(int(layers)):
                        col1, col2 = st.columns(2)
                        ustring = "Enter number of units at hidden layer " + str(i + 1)
                        astring = (
                            "Enter activation function before hidden layer "
                            + str(i + 1)
                        )
                        with col1:
                            ui = int(st.number_input(ustring, key=i, step=1))
                        with col2:
                            ai = st.selectbox(
                                astring, ("", "sigmoid", "relu", "tanh"), key=i
                            )
                        units[i] = ui
                        activation[i] = ai
                    print(activation)
                    if activation[-1] != "" and units[-1] != 0:
                        model = Sequential()
                        for i in range(layers):
                            if i == 0:
                                model.add(
                                    Dense(
                                        units[i],
                                        activation[i],
                                        input_shape=(X_train.shape[1],),
                                    )
                                )
                            else:
                                model.add(Dense(units[i], activation[i]))
                        model.add(Dense(1, "sigmoid"))
                        print(model.summary())

                if st.button("Train"):
                    model.compile(
                        loss=tf.keras.losses.binary_crossentropy,
                        optimizer=tf.keras.optimizers.Adam(lr=0.001),
                        metrics=["accuracy"],
                    )
                    reduce_lr = ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.3,
                        verbose=1,
                        patience=10,
                        min_lr=0.0000000001,
                    )
                    early_stopping_cb = EarlyStopping(
                        patience=10, restore_best_weights=True
                    )
                    my_bar = st.progress(0)

                    class CustomCallback(tf.keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            global percent
                            percent += 1 / epochs
                            if percent > 1:
                                percent = 1
                            my_bar.progress(percent)

                    history_1 = model.fit(
                        X_train,
                        y_train,
                        epochs=epochs,
                        validation_split=0.2,
                        callbacks=[early_stopping_cb, reduce_lr, CustomCallback()],
                    )

                    plot_loss_curves(history_1)

                    nn_f1 = evaluate("nn", model, X_train, y_train, X_test, y_test)
                    st.write("Model trained with an F1 score of", nn_f1)


n_cpus = multiprocessing.cpu_count()

choice = st.selectbox(
    "Which models do you want to run?",
    [
        "",
        "Feature Engineered",
        "Feature Engineered + Scaling",
        "Feature Engineered + SMOTE",
        "Feature Engineered + SMOTE + Scaling",
        "Feature Engineered + PCA  + SMOTE",
    ],
)

if len(choice) > 0:
    if choice == "Feature Engineered":
        d = {
            "Logistic Regression": {"C": 1.0, "solver": "liblinear"},
            "SVM": {"C": 1.0, "gamma": 0.001, "kernel": "rbf"},
            "Random Forest": {
                "max_features": 0.25,
                "min_samples_split": 6,
                "n_estimators": 250,
            },
            "KNN": {"n": 8},
            "xgboost": {
                "booster": "gbtree",
                "colsample_bytree": 0.6,
                "learning_rate": 0.5,
                "max_depth": 2,
                "min_child_weight": 0.001,
                "n_estimators": 9,
            },
            "Lightboost": {
                "colsample_bytree": 0.5,
                "learning_rate": 0.2,
                "max_depth": 5,
                "n_estimators": 50,
                "num_leaves": 4,
                "reg_lambda": 15,
                "scale_pos_weight": 3,
                "subsample": 0.9,
            },
        }
        run(X_train, y_train, X_test, y_test, d)

    elif choice == "Feature Engineered + Scaling":
        sc = StandardScaler()
        X_train_std = X_train.copy()
        X_test_std = X_test.copy()
        X_train_std[
            ["Tenure Months", "Monthly Charges", "Total Charges"]
        ] = sc.fit_transform(
            np.array(X_train_std[["Tenure Months", "Monthly Charges", "Total Charges"]])
        )
        X_test_std[
            ["Tenure Months", "Monthly Charges", "Total Charges"]
        ] = sc.fit_transform(
            np.array(X_test_std[["Tenure Months", "Monthly Charges", "Total Charges"]])
        )
        d = {
            "Logistic Regression": {"C": 1000.0, "solver": "liblinear"},
            "SVM": {"C": 10, "gamma": 0.01, "kernel": "rbf"},
            "Random Forest": {
                "max_features": 0.25,
                "min_samples_split": 6,
                "n_estimators": 350,
            },
            "KNN": {"n": 8},
            "xgboost": {
                "booster": "gbtree",
                "colsample_bytree": 0.6,
                "learning_rate": 0.5,
                "max_depth": 2,
                "min_child_weight": 0.001,
                "n_estimators": 9,
            },
            "Lightboost": {
                "colsample_bytree": 0.5,
                "learning_rate": 0.2,
                "max_depth": 5,
                "n_estimators": 50,
                "num_leaves": 4,
                "reg_lambda": 15,
                "scale_pos_weight": 3,
                "subsample": 0.9,
            },
        }
        run(X_train_std, y_train, X_test_std, y_test, d)

    elif choice == "Feature Engineered + SMOTE":
        sm = SMOTE(random_state=0)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
        d = {
            "Logistic Regression": {"C": 100.0, "solver": "liblinear"},
            "SVM": {"C": 1.0, "gamma": 0.1, "kernel": "rbf"},
            "Random Forest": {
                "max_features": "sqrt",
                "min_samples_split": 2,
                "n_estimators": 350,
            },
            "KNN": {"n": 8},
            "xgboost": {
                "booster": "gbtree",
                "colsample_bytree": 1,
                "learning_rate": 0.6,
                "max_depth": 4,
                "min_child_weight": 0.001,
                "n_estimators": 9,
            },
            "Lightboost": {
                "colsample_bytree": 0.5,
                "learning_rate": 0.2,
                "max_depth": 7,
                "n_estimators": 100,
                "num_leaves": 11,
                "reg_lambda": 10,
                "scale_pos_weight": 3,
                "subsample": 0.9,
            },
        }
        run(X_train_res, y_train_res, X_test, y_test, d)

    elif choice == "Feature Engineered + SMOTE + Scaling":
        sc = StandardScaler()
        X_train_std = X_train.copy()
        X_test_std = X_test.copy()
        X_train_std[
            ["Tenure Months", "Monthly Charges", "Total Charges"]
        ] = sc.fit_transform(
            np.array(X_train_std[["Tenure Months", "Monthly Charges", "Total Charges"]])
        )
        X_test_std[
            ["Tenure Months", "Monthly Charges", "Total Charges"]
        ] = sc.fit_transform(
            np.array(X_test_std[["Tenure Months", "Monthly Charges", "Total Charges"]])
        )
        sm = SMOTE(random_state=0)
        X_train_std_res, y_train_std_res = sm.fit_resample(X_train_std, y_train.ravel())
        d = {
            "Logistic Regression": {"C": 0.1, "solver": "liblinear"},
            "SVM": {"C": 1000, "gamma": 0.0001, "kernel": "rbf"},
            "Random Forest": {
                "max_features": "sqrt",
                "min_samples_split": 6,
                "n_estimators": 350,
            },
            "KNN": {"n": 8},
            "xgboost": {
                "booster": "gbtree",
                "colsample_bytree": 0.9,
                "learning_rate": 0.5,
                "max_depth": 4,
                "min_child_weight": 0.001,
                "n_estimators": 9,
            },
            "Lightboost": {
                "colsample_bytree": 0.5,
                "learning_rate": 0.2,
                "max_depth": 5,
                "n_estimators": 100,
                "num_leaves": 10,
                "reg_lambda": 25,
                "scale_pos_weight": 3,
                "subsample": 0.9,
            },
        }
        run(X_train_std_res, y_train_std_res, X_test_std, y_test, d)
    else:
        sc = StandardScaler()
        X_train_std = X_train.copy()
        X_test_std = X_test.copy()
        X_train_std[
            ["Tenure Months", "Monthly Charges", "Total Charges"]
        ] = sc.fit_transform(
            np.array(X_train_std[["Tenure Months", "Monthly Charges", "Total Charges"]])
        )
        X_test_std[
            ["Tenure Months", "Monthly Charges", "Total Charges"]
        ] = sc.fit_transform(
            np.array(X_test_std[["Tenure Months", "Monthly Charges", "Total Charges"]])
        )
        pca = PCA(0.8)
        pca.fit(X_train_std)
        X_train_pca = pca.transform(X_train_std)
        X_test_pca = pca.transform(X_test_std)
        sm = SMOTE(random_state=10)
        X_train_std_res, y_train_std_res = sm.fit_resample(X_train_pca, y_train.ravel())
        d = {
            "Logistic Regression": {"C": 0.1, "solver": "liblinear"},
            "SVM": {"C": 1000, "gamma": 0.0001, "kernel": "rbf"},
            "Random Forest": {
                "max_features": "sqrt",
                "min_samples_split": 6,
                "n_estimators": 350,
            },
            "KNN": {"n": 8},
            "xgboost": {
                "booster": "gbtree",
                "colsample_bytree": 0.9,
                "learning_rate": 0.5,
                "max_depth": 4,
                "min_child_weight": 0.001,
                "n_estimators": 9,
            },
            "Lightboost": {
                "colsample_bytree": 0.5,
                "learning_rate": 0.2,
                "max_depth": 5,
                "n_estimators": 100,
                "num_leaves": 10,
                "reg_lambda": 25,
                "scale_pos_weight": 3,
                "subsample": 0.9,
            },
        }
        run(X_train_std_res, y_train_std_res, X_test_pca, y_test, d)
