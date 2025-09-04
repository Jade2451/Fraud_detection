import pandas as pd
from pathlib import Path
import argparse

# --- Configuration: Define file paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = BASE_DIR / "results" / "predictions" / "predictions.csv"
REPORT_PATH = BASE_DIR / "results" / "reports" / "risk_dashboard.csv"

def generate_report(predictions_df: pd.DataFrame, risk_threshold: float = 0.5) -> pd.DataFrame:

    print("Generating risk dashboard...")
    high_risk_df = predictions_df[predictions_df['fraud_prediction'] == 1].copy()
    
    if high_risk_df.empty:
        print(" No transactions were flagged as fraudulent. Report will be empty.")
        return high_risk_df

    report_cols = {
        'Time': 'Transaction Time (s)',
        'Amount': 'Transaction Amount',
        'user_id': 'User ID',
        'fraud_prediction': 'Flagged as Fraud',
        'fraud_probability': 'Fraud Probability (%)'
    }
    # Keep original V-columns for further investigation if needed
    v_cols = [col for col in high_risk_df.columns if col.startswith('V')]
    final_cols = list(report_cols.keys()) + v_cols
    
    report_df = high_risk_df[final_cols].rename(columns=report_cols)

    report_df['Fraud Probability (%)'] = (report_df['Fraud Probability (%)'] * 100).round(2)
    report_df['Transaction Amount'] = report_df['Transaction Amount'].round(2)
    
    report_df = report_df.sort_values(by='Fraud Probability (%)', ascending=False)
    
    print(f" Generated report with {len(report_df)} high-risk transactions.")
    return report_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Risk Dashboard Generation Script")
    parser.add_argument(
        '--input_file',
        type=str,
        default=PREDICTIONS_PATH,
        help=f"Path to the input CSV with predictions. Defaults to: {PREDICTIONS_PATH}"
    )
    args = parser.parse_args()

    try:
        predictions_data = pd.read_csv(args.input_file)
        risk_report = generate_report(predictions_data)

        # --- 5. Save the Report ---
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        risk_report.to_csv(REPORT_PATH, index=False)
        print(f"\n Risk dashboard saved to {REPORT_PATH}")

        print("\n--- High-Risk Transactions Preview ---")
        print(risk_report.head())

    except FileNotFoundError:
        print(f" Error: Predictions file not found at '{args.input_file}'. Please run the prediction script first.")
    except Exception as e:
        print(f"An error occurred: {e}")