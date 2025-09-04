# Financial Transaction Anomaly Detection Pipeline

This project demonstrates an end-to-end pipeline for detecting fraudulent financial transactions. The system processes raw transaction data, engineers relevant features using SQL, trains a machine learning model, and generates a risk dashboard for analysis.

## Project Overview

The core objective is to build a reproducible and automated system that can identify potentially fraudulent transactions from a stream of financial data. This project showcases skills in data processing, feature engineering, machine learning, and reporting.

**Key Technologies:**
* **Data Storage & Feature Extraction:** SQLite and SQL
* **Model Training & Prediction:** Python (Pandas, Scikit-learn, TensorFlow/Keras)
* **Pipeline Orchestration:** Python scripting
* **Reporting:** Generation of a CSV report for analysis in tools like Excel or Google Sheets.

**The pipeline follows these steps:**
1.  **Data Ingestion:** Raw transaction data (as a CSV) is loaded into a local SQLite database.
2.  **Feature Engineering:** SQL queries are executed against the database to create features indicative of fraud (e.g., transaction frequency, amount deviations).
3.  **Model Training:** A machine learning model (e.g., Logistic Regression, RandomForest, or a Neural Network) is trained on the engineered features.
4.  **Inference & Prediction:** The trained model scores new transactions, assigning a fraud probability to each.
5.  **Reporting:** A risk dashboard (as a CSV file) is generated, highlighting flagged transactions for review.



##  Setup and Usage

1. Clone the repository:
```bash
git clone [https://github.com/your-username/fraud-detection-pipeline.git](https://github.com/your-username/fraud-detection-pipeline.git)
cd fraud-detection-pipeline
```
2. Download the data:
Download the Credit Card Fraud Detection dataset from Kaggle and place the `creditcard.csv` file inside the `data/raw/` directory.
Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

3. Create a virtual environment and install dependencies:

```Bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```
4. Run the full pipeline:

Execute the main pipeline script. This will perform all steps from data ingestion to final report generation.

```
python main.py
```
The final trained model will be saved in the models/ directory, and the risk dashboard report will be generated in `results/reports/`.

## Repository Structure
```
├── data/
│   ├── raw/
│   │   └── creditcard.csv  (Download from Kaggle)
│   └── processed/          (Processed data will be saved here)
├── models/                 # Trained models will be saved here
├── notebooks/              # Jupyter notebooks for exploratory data analysis (EDA)
│   └── 1-EDA.ipynb
├── results/
│   ├── reports/            # Output risk dashboards (CSV)
│   └── figures/            # Saved plots (e.g., confusion matrix)
├── sql/                    # SQL scripts for feature engineering
│   └── feature_engineering.sql
├── src/                    # Source code for the pipeline
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── predict.py
│   └── reporting.py
├── .gitignore
├── main.py                 # Main script to run the entire pipeline
├── README.md
└── requirements.txt
```

##  Results

The model is evaluated on its ability to distinguish fraudulent from legitimate transactions. The following is a sample performance summary for a `RandomForestClassifier`.

<img width="396" height="196" alt="image" src="https://github.com/user-attachments/assets/8915dd87-a075-4db0-b132-23b09dc99078" />


*Note: Due to the highly imbalanced nature of fraud datasets, precision and recall on the minority class (fraud) are more important metrics than overall accuracy.*

Thoughts:
You can skip this part if you want.
This was a really cool project overall. It helped me understand the end-to-end pipeline of solving a real-world issue while focusing on the pipeline itself. (You might have noticed I only used a simple Random Forest classifier. One, because I love that algorithm in particular. Two, I wanted to focus on the implementation itself)
