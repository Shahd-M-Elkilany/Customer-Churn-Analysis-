# 📊 Customer Churn Prediction Dashboard

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![Scikit-Learn](https://img.shields.io/badge/ML-ScikitLearn-yellow)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

An interactive **Machine Learning web app** built with **Streamlit** for predicting **customer churn** using various models  
(Logistic Regression, Random Forest, and XGBoost). The project includes **data preprocessing**, **feature engineering**,  
**drift detection**, and **MLflow experiment tracking**.

---

## 🚀 Features

- 🧩 **Automated preprocessing & encoding** for Telco Customer Churn dataset  
- 📈 **Model comparison dashboard** (ROC curves, classification reports, feature importance)  
- 🧠 **Supports multiple models** — Logistic Regression, Random Forest, and XGBoost  
- 💾 **Model training and saving** to `/artifacts` folder  
- 🔮 **Single-customer churn prediction** from sidebar input  
- 📊 **Data drift detection** using `alibi-detect` (Kolmogorov–Smirnov test)  
- ⚙️ **MLflow integration** for experiment tracking  

---

## 📦 Folder Structure

```

Customer-Churn-Analysis/
├── app123.py                     # Streamlit web app
├── Finallllll.ipynb              # Jupyter Notebook for data exploration & training
├── artifacts/
│   ├── best_model.pkl            # Saved best-performing model
│   └── final_result.json         # Metadata of best model & metrics
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── data/
└── WA_Fn-UseC_-Telco-Customer-Churn.csv (optional)

````

---

## 🧠 Tech Stack

- **Language:** Python (3.12)
- **Libraries:** Scikit-learn · XGBoost · Pandas · Numpy · Seaborn · Plotly · Matplotlib  
- **Web Framework:** Streamlit  
- **Experiment Tracking:** MLflow  
- **Drift Detection:** Alibi-Detect  

---

## ⚙️ Installation & Setup

Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis
````

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the app locally:

```bash
streamlit run app123.py
```

Then open your browser at 👉 [http://localhost:8501](http://localhost:8501)

---

## 📊 Dataset

The dataset used is the **Telco Customer Churn** dataset from Kaggle:

> [Telco Customer Churn - Kaggle Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

It contains customer demographics, service usage, and churn labels (`Yes` / `No`).

---

## 📈 Model Performance (example results)

| Model               | Accuracy | Precision | Recall | F1   | ROC AUC |
| ------------------- | -------- | --------- | ------ | ---- | ------- |
| Logistic Regression | 0.82     | 0.79      | 0.75   | 0.77 | 0.86    |
| Random Forest       | 0.85     | 0.83      | 0.81   | 0.82 | 0.90    |
| XGBoost             | 0.87     | 0.84      | 0.83   | 0.83 | 0.92    |

---

## 🧪 MLflow Tracking

Track experiments and models locally by running:

```bash
mlflow ui
```

Then visit:
👉 [http://localhost:5000](http://localhost:5000)
to see model parameters, metrics, and artifacts.

---

## 🧮 Drift Detection Example

Data drift is detected using **Alibi Detect** (`KSDrift` test):

```python
from alibi_detect.cd import KSDrift
cd = KSDrift(X_train, p_val=0.05)
result = cd.predict(X_test)
print(result)
```

Output:

```
{'data': {'is_drift': 0, 'p_val': [0.86, 0.99, 0.86, 0.73], 'threshold': 0.0125}}
```

✅ `is_drift: 0` means **no drift detected**.

---

## 🌐 Deployment

To deploy your Streamlit app online:

1. Push this repository to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io)
3. Connect your GitHub and select this repo
4. Set the entry point as:

   ```
   app123.py
   ```
5. Streamlit will auto-install requirements and host your app for free 🎉

---

## 👩‍💻 Author

**Shahd Elkilany**
AI & Data Science Intern @ DEPI | Passionate about AI & Space Tech
📧 shahdelkilanyy@gmail.com

---

## 🪪 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute with attribution.

---

⭐ **If you found this useful, don’t forget to give it a star on GitHub!**
