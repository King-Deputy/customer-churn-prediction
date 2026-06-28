
# Customer Churn Prediction

> An end-to-end machine learning project that predicts customer churn using the IBM Telco Customer Churn dataset. This repository demonstrates a complete machine learning workflow, including data exploration, preprocessing, feature engineering, model development, evaluation, explainability, deployment, and reproducible software engineering practices.

![GitHub release](https://img.shields.io/github/v/release/King-Deputy/customer-churn-prediction?include_prereleases)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Project Overview

Customer churn is one of the most important business challenges for subscription-based companies. Predicting which customers are likely to leave enables organizations to implement proactive retention strategies, reduce customer attrition, and improve long-term profitability.

This project develops an end-to-end machine learning pipeline using the IBM Telco Customer Churn dataset. In addition to predictive modelling, the project emphasizes software engineering best practices such as modular code organization, reproducibility, version control, documentation, and deployment.

This repository is designed as both a portfolio project and a reusable template for future machine learning projects.

---

## Project Objectives

- Understand customer churn behaviour through exploratory analysis.
- Build a robust data preprocessing pipeline.
- Engineer predictive features.
- Train and compare multiple machine learning algorithms.
- Evaluate models using appropriate classification metrics.
- Interpret model predictions using Explainable AI (SHAP).
- Deploy the final model using Streamlit.
- Demonstrate professional Git and GitHub workflows.

---

## Dataset

**Dataset:** IBM Telco Customer Churn Dataset

Dataset Summary

| Attribute | Value |
|------------|-------|
| Records | 7,043 |
| Features | 33 |
| Target Variable | Churn Label |
| Churned Customers | 1,869 |
| Retained Customers | 5,174 |

The dataset contains customer demographic information, subscribed services, billing information, contract details, payment methods, customer lifetime value, and churn status.

---

## Repository Structure

```text
customer-churn-prediction/
│
├── app/
├── config/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── README.md
│
├── docs/
├── figures/
├── models/
├── notebooks/
├── outputs/
├── src/
├── tests/
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

---
```
## Technology Stack

| Category             | Technologies          |
| -------------------- | --------------------- |
| Programming Language | Python 3.12           |
| Data Analysis        | Pandas, NumPy         |
| Visualization        | Matplotlib, Plotly    |
| Machine Learning     | Scikit-learn, XGBoost |
| Explainability       | SHAP                  |
| Deployment           | Streamlit             |
| Version Control      | Git & GitHub          |
| IDE                  | Visual Studio Code    |

---

## Project Progress

### Completed

* [x] Repository setup
* [x] Project structure
* [x] Python virtual environment
* [x] Data acquisition
* [x] Data loading module
* [x] Data understanding
* [x] Exploratory data analysis

### Upcoming

* [ ] Data preprocessing
* [ ] Feature engineering
* [ ] Baseline machine learning models
* [ ] Advanced machine learning models
* [ ] Hyperparameter tuning
* [ ] Model evaluation
* [ ] Explainable AI (SHAP)
* [ ] Streamlit deployment
* [ ] Unit testing
* [ ] Documentation refinement

---

## Current Release

| Version | Status      |
| ------- | ----------- |
| v0.1.0  | Pre-release |

Current milestone:

* Repository initialization
* IBM Telco dataset integration
* Data Understanding
* Exploratory Data Analysis

---

## Installation

Clone the repository

```bash
git clone https://github.com/King-Deputy/customer-churn-prediction.git
```

Navigate to the project directory

```bash
cd customer-churn-prediction
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

### Windows (Git Bash)

```bash
source .venv/Scripts/activate
```

### Windows (PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Launch Jupyter Lab

```bash
jupyter lab
```

Run the notebooks in the following order:

1. `01_data_understanding.ipynb`
2. `02_exploratory_data_analysis.ipynb`

---

## Results

The project currently includes:

* Dataset profiling
* Missing value analysis
* Duplicate detection
* Target variable analysis
* Numerical feature exploration
* Categorical feature exploration
* Correlation analysis
* Business insights

Model development is currently in progress.

---

## Roadmap

* Data preprocessing pipeline
* Feature engineering
* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* Hyperparameter optimisation
* SHAP explainability
* Streamlit deployment
* Model monitoring
* Docker support

---

## Future Improvements

* Continuous Integration (GitHub Actions)
* Docker containerization
* MLflow experiment tracking
* Automated model retraining
* Cloud deployment

---

## License

This project is licensed under the MIT License.

---

## Author

**Akogun Sodiq**

GitHub: https://github.com/King-Deputy

---

## Acknowledgements

* IBM Telco Customer Churn Dataset
* Scikit-learn
* XGBoost
* SHAP
* Streamlit

---

⭐ If you find this repository useful, consider starring it to support the project.

```
```
