import pandas as pd
from pathlib import Path
import os

# ---- base folder ----
# directory where you have the 6 starting folders
DATA_DIR = Path("data/processed/combined_DS/v3/30_frame_segments")
# directory where the code automatically created 6 new folders with the extracted data
EXTRACTION_DIR = Path("data/processed/combined_DS/v3/1_frame_segments")

# ---- folders iteration ----
for exercise in os.listdir(DATA_DIR):
    path_exercise = os.path.join(DATA_DIR, exercise)
    # if not a directory jump it
    if not os.path.isdir(path_exercise):
        continue

    # ---- output folder generation ----
    output_sottocartella = os.path.join(EXTRACTION_DIR, exercise)
    os.makedirs(output_sottocartella, exist_ok=True)

    # ---- csv iteration ----
    for nome_csv in os.listdir(path_exercise):
        if nome_csv.endswith('.csv'):
            path_csv = os.path.join(path_exercise, nome_csv)
            # csv reading
            df = pd.read_csv(path_csv)
            # drop last column
            df = df.iloc[:, :-1]
            # extraction and saving
            for i, row in df.iterrows():
                nome_base = os.path.splitext(nome_csv)[0]
                nome_output = f"{nome_base}_{i+1}.csv"
                path_output = os.path.join(output_sottocartella, nome_output)
                # saving single row without the header
                row.to_frame().T.to_csv(path_output, index=False, header=False)