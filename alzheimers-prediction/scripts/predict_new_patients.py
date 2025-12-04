# scripts/predict_new_patients.py

import pandas as pd
import joblib
import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_new_patients.py <path_to_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    if not os.path.exists(input_csv):
        print(f"Error: File {input_csv} not found.")
        sys.exit(1)

    # Load new patient data
    new_data = pd.read_csv(input_csv)

    # Load the saved pipeline
    pipeline_path = os.path.join(os.path.dirname(__file__), '../models/alzheimers_pipeline.pkl')
    pipeline = joblib.load(pipeline_path)

    # Make predictions
    preds = pipeline.predict(new_data)
    probs = pipeline.predict_proba(new_data)[:, 1]

    # Create output DataFrame
    output_df = new_data.copy()
    output_df['Prediction'] = preds
    output_df['Probability'] = probs

    # Save predictions
    output_csv = os.path.splitext(os.path.basename(input_csv))[0] + "_predictions.csv"
    output_df.to_csv(output_csv, index=False)

    print(f"Predictions saved to {output_csv}")
    print(output_df[['Prediction', 'Probability']])

if __name__ == "__main__":
    main()
