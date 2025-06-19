import os
from pathlib import Path

def lowercase_paths(root_dir: str):
    """Recurse under root_dir and rename every folder & file to lowercase."""
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # files
        for filename in filenames:
            old_path = os.path.join(dirpath, filename)
            filename = filename.lower()
            filename = filename.replace(' ', '_')
            new_name = filename.replace('-', '_')
            new_path = os.path.join(dirpath, new_name)
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"Renamed file: {old_path} -> {new_path}")

        # directories
        for dirname in dirnames:
            old_dir = os.path.join(dirpath, dirname)
            dirname = dirname.lower()
            dirname = dirname.replace(' ', '_')
            new_dirname = dirname.replace('-', '_')
            new_dir = os.path.join(dirpath, new_dirname)
            if old_dir != new_dir:
                os.rename(old_dir, new_dir)
                print(f"Renamed dir : {old_dir} -> {new_dir}")

if __name__ == "__main__":
    # Compute the project root, then locate your data folder
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # ../.. from src/helpers
    # target = project_root / "data" / "processed" / "own_DS" / "30_frame_segments"
    target = r"C:\Users\hannes\.cache\kagglehub\datasets\philosopher0808\gym-workoutexercises-video\versions\1\processed_data_hannes\data-btc"
    lowercase_paths(str(target))
