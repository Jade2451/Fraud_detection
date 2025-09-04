import sys
from pathlib import Path

# Add the source directory to the Python path to allow for module imports
sys.path.append(str(Path(__file__).resolve().parent))

from src.feature_engineering import engineer_features
from src.model_training import train_model
from src.predict import make_predictions
from src.report import generate_report
import pandas as pd

# Define file paths for inter-script communication
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "featured_transactions.csv"
PREDICTIONS_PATH = BASE_DIR / "results" / "predictions" / "predictions.csv"
REPORT_PATH = BASE_DIR / "results" / "reports" / "risk_dashboard.csv"


def main():
    print("=================================================")
    print("      Fraud Detection Pipeline - STARTED         ")
    print("=================================================\n")

    print("--- STEP 1: Running Feature Engineering ---")
    engineer_features()
    print("--- STEP 1: COMPLETED ---\n")

    print("--- STEP 2: Running Model Training ---")
    train_model()
    print("--- STEP 2: COMPLETED ---\n")
    
    print("--- STEP 3: Generating Batch Predictions ---")
    try:
        data_for_prediction = pd.read_csv(PROCESSED_DATA_PATH)
        predictions_df = make_predictions(data_for_prediction)
        
        if predictions_df is not None:
            PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
            predictions_df.to_csv(PREDICTIONS_PATH, index=False)
            print(f"Predictions saved to {PREDICTIONS_PATH}")
        else:
            raise Exception("Prediction script failed to return a DataFrame.")
        
        print("--- STEP 3: COMPLETED ---\n")

    except FileNotFoundError:
        print(f" Error in Step 3: Processed data file not found at '{PROCESSED_DATA_PATH}'. Cannot run predictions.")
        return
    except Exception as e:
        print(f" An error occurred during the prediction step: {e}")
        return

    # --- Step 4: Reporting ---
    print("--- STEP 4: Generating Risk Dashboard Report ---")
    try:
        predictions_data = pd.read_csv(PREDICTIONS_PATH)
        risk_report = generate_report(predictions_data)
        
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        risk_report.to_csv(REPORT_PATH, index=False)
        print(f"Risk report saved to {REPORT_PATH}")
        print("--- STEP 4: COMPLETED ---\n")

    except FileNotFoundError:
        print(f" Error in Step 4: Predictions file not found at '{PREDICTIONS_PATH}'. Cannot generate report.")
        return
    except Exception as e:
        print(f" An error occurred during the reporting step: {e}")
        return

    print("=================================================")
    print("      Fraud Detection Pipeline - FINISHED        ")
    print("=================================================\n")

if __name__ == '__main__':
    main()