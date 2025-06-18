import os
from pathlib import Path

def delete_flipped_files():
    # Path to the directory
    base_path = Path("data/processed/combined_DS/v1/30_frame_segments")
    
    # Get all CSV files with 'flipped' in their name
    flipped_files = list(base_path.rglob("*flipped*.csv"))
    
    if not flipped_files:
        print("No flipped files found!")
        return
    
    # Print files that will be deleted
    print(f"Found {len(flipped_files)} files to delete:")
    for file in flipped_files:
        print(f"- {file}")
    
    # Ask for confirmation
    response = input("\nDo you want to proceed with deletion? (yes/no): ")
    
    if response.lower() == 'yes':
        deleted_count = 0
        for file in flipped_files:
            try:
                file.unlink()
                deleted_count += 1
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")
        
        print(f"\nSuccessfully deleted {deleted_count} out of {len(flipped_files)} files")
    else:
        print("Deletion cancelled")

if __name__ == "__main__":
    delete_flipped_files()