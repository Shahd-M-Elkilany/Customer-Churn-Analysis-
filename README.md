# Customer Churn Prediction

Exploratory analysis and classification models for the Telco Customer Churn
dataset (7,043 customers, 21 columns), with a Streamlit app for training and
comparing models on an uploaded CSV.

## What is here

| File | Contents |
| --- | --- |
| `churn_analysis.ipynb` | Data cleaning, exploratory analysis, feature scaling and encoding, and a grid search over Logistic Regression and Random Forest with three scaling options each |
| `app.py` | Streamlit app: upload the CSV, choose features, train Logistic Regression, Random Forest and XGBoost, and compare ROC curves on a held-out 20% split |
| `requirements.txt` | Python dependencies |

The notebook's grid search uses 5-fold stratified cross-validation scored by
ROC AUC, and selects a Random Forest (200 trees, max depth 10) with min-max
scaling.

## Running it

```bash
git clone https://github.com/Shahd-M-Elkilany/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis
pip install -r requirements.txt
mkdir artifacts
streamlit run app.py
```

Then upload the dataset CSV in the browser. Download it from
[Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## Limitations

- The summary metrics in the notebook are computed on the same data the
  models were fitted on, so they overstate performance. The Streamlit app
  evaluates on a held-out split and is the better guide.
- The MLflow tracking and drift-detection cells at the end of the notebook
  are standalone demonstrations on the Iris dataset; they are not yet wired
  to the churn models.

## License

MIT
