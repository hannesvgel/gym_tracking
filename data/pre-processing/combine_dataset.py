import os
import shutil

folder_names = ['bench_press', 'lat_machine', 'pull_up', 'push_up', 'split_squat', 'squat']
output_dir = r'C:\Users\hannes\01_Code\02_master_rci\03_Semester_PoliMi\02_nearables_lab\gym_tracking\data\processed\combined_DS\v2\30_frame_segments'

db_folders = {
    'k': r'C:\Users\hannes\01_Code\02_master_rci\03_Semester_PoliMi\02_nearables_lab\gym_tracking\data\processed\kaggle_DS\30_frame_segments',
    'o': r'C:\Users\hannes\01_Code\02_master_rci\03_Semester_PoliMi\02_nearables_lab\gym_tracking\data\processed\own_DS\30_frame_segments\vertical',
    'f3': r'C:\Users\hannes\01_Code\02_master_rci\03_Semester_PoliMi\02_nearables_lab\gym_tracking\data\processed\fit3D_DS\30_frame_segments'
}

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

for folder_name in folder_names:
    target_folder = os.path.join(output_dir, folder_name)
    os.makedirs(target_folder, exist_ok=True)
    count = 0
    for key, base_path in db_folders.items():
        source_folder = os.path.join(base_path, folder_name)
        if not os.path.exists(source_folder):
            print(f"Warning: {source_folder} does not exist.")
            continue

        for filename in os.listdir(source_folder):
            if filename.endswith('.csv'):
                src_file = os.path.join(source_folder, filename)
                dst_filename = f"{folder_name}_{key}_{count}.csv"
                dst_file = os.path.join(target_folder, dst_filename)
                shutil.copy2(src_file, dst_file)
                count += 1
        