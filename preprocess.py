import os
import librosa
import numpy as np

osu_dir = "organized/osu"
mp3_dir = "organized/mp3"

def extract_audio(mp3_path):
    print(f"Processing MP3: {mp3_path}")  # Debug: Which file fails?
    try:
        y, sr = librosa.load(mp3_path)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        energy = librosa.feature.rms(y=y)[0]
        return beat_times, energy[:len(beat_times)]
    except Exception as e:
        print(f"Error loading {mp3_path}: {e}")
        return None, None  # Return None to skip this file

def parse_osu(osu_path):
    od = float(os.path.basename(osu_path).split("_OD")[1].replace(".osu", ""))
    hit_objs = []
    with open(osu_path, 'r', encoding='utf-8') as f:
        in_hit_objs = False
        for line in f:
            if line == "[HitObjects]":  # Fixed: line.strip() not needed here
                in_hit_objs = True
            elif line.startswith("["):
                break
            elif in_hit_objs and line:
                parts = line.split(",")
                x, y, time = int(parts[0]), int(parts[1]), int(parts[2])
                type_field = int(parts[3])
                obj_type = 0
                new_combo = 0
                if type_field & 1:  # Circle
                    obj_type = 1
                elif type_field & 2:  # Slider
                    obj_type = 2
                elif type_field & 8:  # Spinner
                    obj_type = 3
                if type_field & 4:  # New combo
                    new_combo = 1
                hitsound = int(parts[4]) if len(parts) > 4 else 0
                hit_objs.append((time, x, y, obj_type, new_combo, hitsound))
    return hit_objs, od

def prepare_data():
    X_audio = []
    X_difficulty = []
    y_hit_objs = []

    for mp3_file in os.listdir(mp3_dir):
        if not mp3_file.endswith(".mp3"):
            continue
        
        song_id = mp3_file.replace("song", "").replace(".mp3", "")
        mp3_path = os.path.join(mp3_dir, mp3_file)
        beat_times, energy = extract_audio(mp3_path)
        
        # Skip if audio extraction failed
        if beat_times is None or energy is None:
            continue

        for osu_file in os.listdir(osu_dir):
            if osu_file.startswith(f"song{song_id}_OD"):
                osu_path = os.path.join(osu_dir, osu_file)
                hit_objs, od = parse_osu(osu_path)
                X = np.column_stack((beat_times, energy))
                y = np.array(hit_objs, dtype=np.float32)
                min_len = min(len(X), len(y))
                X_audio.append(X[:min_len])
                X_difficulty.append(np.full(min_len, od))
                y_hit_objs.append(y[:min_len])
    
    return X_audio, X_difficulty, y_hit_objs

if __name__ == "__main__":
    X_audio, X_difficulty, y_hit_objs = prepare_data()
    print(f"Processed {len(X_audio)} beatmaps")