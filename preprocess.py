import os
import librosa
import numpy as np

osu_dir = "organized/osu"
mp3_dir = "organized/mp3"

def extract_audio(mp3_path):
    print(f"Processing MP3: {mp3_path}")
    try:
        y, sr = librosa.load(mp3_path)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        energy = librosa.feature.rms(y=y)[0]
        print(f"Beat times length: {len(beat_times)}, Energy length: {len(energy)}")
        if len(beat_times) == 0:
            print(f"Skipping {mp3_path}: No beats detected")
            return None, None
        return beat_times, energy[:len(beat_times)]
    except Exception as e:
        print(f"Error loading {mp3_path}: {e}")
        return None, None

def parse_osu(osu_path):
    od = float(os.path.basename(osu_path).split("_OD")[1].replace(".osu", ""))
    hit_objs = []
    with open(osu_path, 'r', encoding='utf-8') as f:
        in_hit_objs = False
        for line in f:
            line = line.strip()
            if line == "[HitObjects]":
                in_hit_objs = True
            elif in_hit_objs and line.startswith("["):
                break
            elif in_hit_objs and line:
                parts = line.split(",")
                x, y, time = int(parts[0]), int(parts[1]), int(parts[2])
                type_field = int(parts[3])
                obj_type = 0
                new_combo = 0
                if type_field & 1: obj_type = 1
                elif type_field & 2: obj_type = 2
                elif type_field & 8: obj_type = 3
                if type_field & 4: new_combo = 1
                hitsound = int(parts[4]) if len(parts) > 4 else 0
                hit_objs.append((time, x, y, obj_type, new_combo, hitsound))
    print(f"Parsed {osu_path}: {len(hit_objs)} hit objects")
    return hit_objs, od

def prepare_data():
    X_audio = []
    X_difficulty = []
    y_hit_objs = []
    max_time = 120000  # Adjust based on your longest map

    for mp3_file in os.listdir(mp3_dir):
        if not mp3_file.endswith(".mp3"):
            continue
        
        song_id = mp3_file.replace("song", "").replace(".mp3", "")
        mp3_path = os.path.join(mp3_dir, mp3_file)
        beat_times, energy = extract_audio(mp3_path)
        
        if beat_times is None or energy is None:
            continue

        max_beat = max(beat_times)
        X = np.column_stack((
            beat_times / max_beat,  # Normalize to [0,1]
            energy / np.max(energy)  # Ensure [0,1]
        ))

        for osu_file in os.listdir(osu_dir):
            if osu_file.startswith(f"song{song_id}_OD"):
                osu_path = os.path.join(osu_dir, osu_file)
                hit_objs, od = parse_osu(osu_path)
                y = np.array(hit_objs, dtype=np.float32)
                y[:, 0] /= max_time  # Normalize time
                y[:, 1] /= 512       # Normalize x
                y[:, 2] /= 384       # Normalize y
                y[:, 5] /= 255       # Normalize hitsound
                
                min_len = min(len(X), len(y))
                X_audio.append(X[:min_len])
                X_difficulty.append(np.full(min_len, od / 10))  # Normalize OD
                y_hit_objs.append(y[:min_len])
    
    return X_audio, X_difficulty, y_hit_objs

if __name__ == "__main__":
    X_audio, X_difficulty, y_hit_objs = prepare_data()
    print(f"Processed {len(X_audio)} beatmaps")