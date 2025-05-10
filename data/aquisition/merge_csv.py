import os
import pandas as pd

# --- CARTELLA DEI CSV DA UNIRE ---
INPUT_FOLDER = "pose_csv_segments"
OUTPUT_FILE = "dataset_unificato.csv"

# --- RACCOLTA CSV ---
all_dataframes = []

for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".csv"):
        csv_path = os.path.join(INPUT_FOLDER, filename)
        df = pd.read_csv(csv_path)

        # Verifica presenza colonna label
        if 'label' in df.columns:
            all_dataframes.append(df)
        else:
            print(f"⚠️ Il file {filename} non contiene una colonna 'label' e verrà ignorato.")

# --- UNISCI TUTTO ---
if all_dataframes:
    final_df = pd.concat(all_dataframes, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Dataset unificato salvato in: {OUTPUT_FILE}")
else:
    print("❌ Nessun CSV valido trovato nella cartella.")
