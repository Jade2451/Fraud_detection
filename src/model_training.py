import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE

# --- Configuration: Define file paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "featured_transactions.csv"
MODEL_OUTPUT_PATH = BASE_DIR / "models" / "fraud_detector.joblib"
FEATURE_LIST_PATH = BASE_DIR / "models" / "feature_list.joblib"

def train_model():
    """
    Orchestrates the model training, evaluation, and saving process.
    """
    print("Starting model training process...")

    # --- 1. Load Data ---
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        print(f" Feature-engineered data loaded from {PROCESSED_DATA_PATH}")
    except FileNotFoundError:
        print(f" Error: Processed data file not found. Please run the feature engineering script first.")
        return

    # --- 2. Define Features (X) and Target (y) ---
    # The target variable is 'Class' (1 for fraud, 0 for normal)
    y = df['Class']
    
    # Drop non-feature columns. 'Time' can be debated, but for this model, we'll use
    # the engineered time-based features instead. 'user_id' was for feature generation only.
    X = df.drop(columns=['Class', 'Time', 'user_id', 'index'])
    
    # Save the list of feature names used for training
    feature_names = X.columns.tolist()
    
    print(f"Defined {len(feature_names)} features and target variable.")

    # --- 3. Split Data into Training and Testing Sets ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Data split into training and testing sets.")

    # --- 4. Handle Class Imbalance with SMOTE ---
    # SMOTE should ONLY be applied to the training data to avoid data leakage.
    print("Applying SMOTE to handle class imbalance on the training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f" SMOTE applied. Original training samples: {len(y_train)}, Resampled: {len(y_train_resampled)}")

    # --- 5. Train the RandomForest Model ---
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train_resampled, y_train_resampled)
    print(" Model training complete.")

    # --- 6. Evaluate the Model ---
    print("\n--- Model Evaluation on Test Set ---")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud']))
    
    # --- 7. Save the Model ---
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    joblib.dump(feature_names, FEATURE_LIST_PATH)
    print(f"\n Trained model saved to {MODEL_OUTPUT_PATH}")
    print(f" Feature list saved to {FEATURE_LIST_PATH}")


if __name__ == '__main__':
    train_model()