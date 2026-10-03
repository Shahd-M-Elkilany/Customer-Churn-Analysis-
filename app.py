import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt

# Streamlit UI config
st.set_page_config(page_title="Churn Prediction App", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")

# Load saved model if available
try:
    best_model = joblib.load("artifacts/best_model.pkl")
    st.sidebar.success("✅ Loaded saved model from artifacts/")
except Exception:
    best_model = None
    st.sidebar.warning("⚠️ No saved model found. Upload dataset and retrain below.")

# File uploader
uploaded_file = st.file_uploader("Upload Telco CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    target = st.selectbox("Select target variable", df.columns)
    features = st.multiselect("Select features", df.columns.drop(target))

    if st.button("Run Pipeline") and features:
        num_features = df[features].select_dtypes(include=np.number).columns.tolist()
        cat_features = list(set(features) - set(num_features))

        X = df[num_features + cat_features]
        y = df[target].map({'No': 0, 'Yes': 1}).fillna(0)

        from sklearn.model_selection import train_test_split, GridSearchCV
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        numeric_transformer = Pipeline([('scaler', StandardScaler())])
        categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])

        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }

        fig, ax = plt.subplots()

        for name, model in models.items():
            clf = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
            clf.fit(X_train, y_train)
            y_proba = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.2f})')

        ax.plot([0, 1], [0, 1], 'k--')
        ax.legend()
        st.pyplot(fig)

        # Save best model (Random Forest for example)
        best_model = RandomForestClassifier(random_state=42)
        pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', best_model)])
        pipeline.fit(X, y)
        joblib.dump(pipeline, "artifacts/best_model.pkl")
        st.success("✅ Best model retrained and saved to artifacts/")

# Sidebar — single customer prediction
st.sidebar.header("🔮 Predict on New Customer")
if best_model:
    new_data = {}
    example_input = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in example_input:
        new_data[col] = st.sidebar.number_input(col, value=50.0)
    new_df = pd.DataFrame([new_data])

    if st.sidebar.button("Predict Churn"):
        pred = best_model.predict(new_df)[0]
        st.sidebar.write(f"Prediction: {'Churn' if pred==1 else 'Not Churn'}")
else:
    st.sidebar.info("Train or load a model first to make predictions.")
