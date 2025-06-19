import moviepy
from pathlib import Path

def batch_trim_videos(input_dir: str, output_dir: str,
                      trim_start: float = 6, trim_end: float = 6,
                      min_remaining: float = 1.0):
    """
    Trim the first `trim_start` and last `trim_end` seconds from each .mp4 in input_dir.
    Skips any video whose duration is less than trim_start + trim_end + min_remaining.
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    print(f"Input: {input_path}\nOutput: {output_path}")
    print(f"Trimming: {trim_start}s start, {trim_end}s end (need ≥{min_remaining}s left)")

    if not input_path.exists():
        print(f"ERROR: Input directory {input_path} does not exist!")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    video_files = list(input_path.glob("*.mp4"))
    # Filter to only include videos with 'push' or 'squat' in the name
    video_files = [v for v in video_files if 'push' in v.name.lower() 
                   or 'squat' in v.name.lower()]
    print(f"Found {len(video_files)} .mp4 files (filtered to push/squat only)")

    if not video_files:
        return

    processed = error = 0
    for idx, video in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] {video.name}")
        try:
            clip = moviepy.VideoFileClip(str(video))
            dur = clip.duration
            print(f"  Duration: {dur:.2f}s")

            if dur < (trim_start + trim_end + min_remaining):
                print("  SKIP: video too short to trim without dropping below minimum length.")
                clip.close()
                continue

            start = trim_start
            end   = dur - trim_end
            print(f"  Trimming to subclip({start:.2f}, {end:.2f}) → {(end-start):.2f}s")

            # Get the folder name before 'videos' and the last part of the input path
            input_path_parts = input_path.parts
            videos_index = None
            for i, part in enumerate(input_path_parts):
                if part == 'videos':
                    videos_index = i
                    break
            
            if videos_index is not None and videos_index > 0:
                folder_before_videos = input_path_parts[videos_index - 1]
                input_path_suffix = f"{folder_before_videos}_{input_path.name}"
            else:
                input_path_suffix = input_path.name
            
            sub = clip.subclipped(start, end)
            # Add the input path suffix to the output filename
            output_filename = f"{video.stem}_{input_path_suffix}{video.suffix}"
            out = output_path / output_filename
            sub.write_videofile(str(out), codec="libx264", audio_codec="aac")
            sub.close()
            clip.close()

            print("Saved:", out)
            processed += 1

        except Exception as e:
            print("ERROR:", e)
            error += 1

    print(f"\nDone — {processed} succeeded, {error} failed out of {len(video_files)}.")

if __name__ == "__main__":

    input_path_list = [
        r"C:\Users\hannes\.cache\fit3D\train\s03\videos\50591643", 
        r"C:\Users\hannes\.cache\fit3D\train\s03\videos\58860488",
        r"C:\Users\hannes\.cache\fit3D\train\s03\videos\60457274",
        r"C:\Users\hannes\.cache\fit3D\train\s03\videos\65906101",
        r"C:\Users\hannes\.cache\fit3D\train\s04\videos\50591643", 
        r"C:\Users\hannes\.cache\fit3D\train\s04\videos\58860488",
        r"C:\Users\hannes\.cache\fit3D\train\s04\videos\60457274",
        r"C:\Users\hannes\.cache\fit3D\train\s04\videos\65906101",
        r"C:\Users\hannes\.cache\fit3D\train\s05\videos\50591643", 
        r"C:\Users\hannes\.cache\fit3D\train\s05\videos\58860488",
        r"C:\Users\hannes\.cache\fit3D\train\s05\videos\60457274",
        r"C:\Users\hannes\.cache\fit3D\train\s05\videos\65906101",
        r"C:\Users\hannes\.cache\fit3D\train\s07\videos\50591643", 
        r"C:\Users\hannes\.cache\fit3D\train\s07\videos\58860488",
        r"C:\Users\hannes\.cache\fit3D\train\s07\videos\60457274",
        r"C:\Users\hannes\.cache\fit3D\train\s07\videos\65906101",
        r"C:\Users\hannes\.cache\fit3D\train\s08\videos\50591643", 
        r"C:\Users\hannes\.cache\fit3D\train\s08\videos\58860488",
        r"C:\Users\hannes\.cache\fit3D\train\s08\videos\60457274",
        r"C:\Users\hannes\.cache\fit3D\train\s08\videos\65906101",
        r"C:\Users\hannes\.cache\fit3D\train\s09\videos\50591643", 
        r"C:\Users\hannes\.cache\fit3D\train\s09\videos\58860488",
        r"C:\Users\hannes\.cache\fit3D\train\s09\videos\60457274",
        r"C:\Users\hannes\.cache\fit3D\train\s09\videos\65906101",
        r"C:\Users\hannes\.cache\fit3D\train\s10\videos\50591643", 
        r"C:\Users\hannes\.cache\fit3D\train\s10\videos\58860488",
        r"C:\Users\hannes\.cache\fit3D\train\s10\videos\60457274",
        r"C:\Users\hannes\.cache\fit3D\train\s10\videos\65906101",
        r"C:\Users\hannes\.cache\fit3D\train\s11\videos\50591643", 
        r"C:\Users\hannes\.cache\fit3D\train\s11\videos\58860488",
        r"C:\Users\hannes\.cache\fit3D\train\s11\videos\60457274",
        r"C:\Users\hannes\.cache\fit3D\train\s11\videos\65906101",
        ]

    for input_path in input_path_list:
        batch_trim_videos(
            input_dir=input_path,
            output_dir=r"C:\Users\hannes\.cache\fit3D\train\cropped_data_hannes"
        )
