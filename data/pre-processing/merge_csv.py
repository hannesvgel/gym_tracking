import os
import pandas as pd

# --- FOLDER OF CSV FILES TO MERGE ---
INPUT_FOLDER = "pose_csv_segments"
OUTPUT_FILE = "unified_dataset.csv"

# --- COLLECT CSV FILES ---
all_dataframes = []

for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".csv"):
        csv_path = os.path.join(INPUT_FOLDER, filename)
        df = pd.read_csv(csv_path)

        # Check for presence of 'label' column
        if 'label' in df.columns:
            all_dataframes.append(df)
        else:
            print(f"File {filename} does not contain a 'label' column and will be ignored.")

# --- MERGE ALL DATAFRAMES ---
if all_dataframes:
    final_df = pd.concat(all_dataframes, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Unified dataset saved to: {OUTPUT_FILE}")
else:
    print("No valid CSV files found in the folder.")
