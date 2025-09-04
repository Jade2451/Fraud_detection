import pandas as pd
from pathlib import Path
import joblib
import argparse

# --- Configuration: Define file paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_detector.joblib"
FEATURE_LIST_PATH = BASE_DIR / "models" / "feature_list.joblib"
DEFAULT_INPUT_PATH = BASE_DIR / "data" / "processed" / "featured_transactions.csv"
OUTPUT_PATH = BASE_DIR / "results" / "predictions" / "predictions.csv"

def make_predictions(input_df: pd.DataFrame) -> pd.DataFrame:
    print("Starting prediction process...")

    try:
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURE_LIST_PATH)
        print(f" Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f" Error: Model or feature list not found. Please run the training script first.")
        return None

    # Ensure all required features are present
    missing_features = set(feature_names) - set(input_df.columns)
    if missing_features:
        print(f" Error: Input data is missing the following required features: {missing_features}")
        return None

    # Ensure the columns are in the same order as during training
    X_predict = input_df[feature_names]
    print(" Input data validated successfully.")

    print("Generating predictions...")
    predictions = model.predict(X_predict)
    probabilities = model.predict_proba(X_predict)[:, 1] # Probability of the positive class (fraud)
    print(" Predictions generated.")


    output_df = input_df.copy()
    output_df['fraud_prediction'] = predictions
    output_df['fraud_probability'] = probabilities

    return output_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fraud Detection Prediction Script")
    parser.add_argument(
        '--input_file',
        type=str,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to the input CSV file with transaction features. Defaults to: {DEFAULT_INPUT_PATH}"
    )
    args = parser.parse_args()

    try:
        input_data = pd.read_csv(args.input_file)
        
        # For demonstration, we'll use the first 1000 rows as "new" data
        sample_data = input_data.head(1000)

        predictions_df = make_predictions(sample_data)

        if predictions_df is not None:
            # --- 5. Save and Display Results ---
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            predictions_df.to_csv(OUTPUT_PATH, index=False)
            print(f"\n Predictions saved to {OUTPUT_PATH}")
            
            print("\n--- Sample Predictions ---")
            # Show original columns plus the new prediction columns
            display_cols = ['Amount', 'fraud_prediction', 'fraud_probability'] + [f for f in predictions_df.columns if 'user' in f]
            print(predictions_df[display_cols].head())

    except FileNotFoundError:
        print(f" Error: Input file not found at '{args.input_file}'")
    except Exception as e:
        print(f"An error occurred: {e}")