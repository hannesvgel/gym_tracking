import os
import glob
import argparse

def rename_files(directory_path, old_string, new_string):
    """
    Rename files in the specified directory by replacing old_string with new_string.
    """
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist!")
        return

    print(f"Processing files in: {directory_path}")
    print(f"Replacing '{old_string}' with '{new_string}'")

    files = glob.glob(os.path.join(directory_path, "*"))
    if not files:
        print("No files found in the directory.")
        return

    print(f"Found {len(files)} files to process.")
    renamed_count = 0

    for file_path in files:
        if os.path.isfile(file_path):
            filename = os.path.basename(file_path)
            if old_string in filename:
                new_filename = filename.replace(old_string, new_string)
                new_file_path = os.path.join(os.path.dirname(file_path), new_filename)

                if new_filename != filename:
                    try:
                        os.rename(file_path, new_file_path)
                        print(f"Renamed: {filename} -> {new_filename}")
                        renamed_count += 1
                    except Exception as e:
                        print(f"Error renaming {filename}: {e}")

    print(f"\nRenaming complete! Total files renamed: {renamed_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rename files by replacing strings in filenames"
    )
    parser.add_argument(
        "--path", "-p",
        required=True,
        help="Path to the directory containing files to rename"
    )
    parser.add_argument(
        "--old", "-o",
        required=True,
        help="String to be replaced in filenames"
    )
    parser.add_argument(
        "--new", "-n",
        required=True,
        help="String to replace the old string with"
    )

    args = parser.parse_args()
    rename_files(args.path, args.old, args.new)

'''
call by e.g.

python data/pre-processing/rename_fit_3D_files.py `
  --path "C:\Users\hannes\.cache\kagglehub\datasets\philosopher0808\gym-workoutexercises-video\versions\1\processed_data_hannes\final_data\lat_machine" `
  --old "lat_pulldown" `
  --new "lat_machine"
'''