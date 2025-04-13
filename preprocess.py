import os
import librosa
import numpy as np

osu_dir = "organized/osu"
mp3_dir = "organized/mp3"

def extract_audio(mp3_path):
    print(f"Processing MP3: {mp3_path}")
    try:
        y, sr = librosa.load(mp3_path, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        if len(beat_times) < 10:
            print(f"Skipping {mp3_path}: Too few beats ({len(beat_times)})")
            return None, None, None
        energy = librosa.feature.rms(y=y)[0]
        beat_frames = np.minimum(beat_frames, len(energy) - 1)
        energy = energy[beat_frames]
        print(f"Beat times length: {len(beat_times)}, Energy length: {len(energy)}")
        print(f"Sample beat_times: {beat_times[:5]}, Sample energy: {energy[:5]}")
        if np.any(np.isnan(energy)) or np.any(np.isinf(energy)):
            print(f"Invalid energy values in {mp3_path}")
            return None, None, None
        return beat_times, energy, tempo
    except Exception as e:
        print(f"Error loading {mp3_path}: {e}")
        return None, None, None

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
                obj_type = 1  # Default to circle
                new_combo = 0
                if type_field & 1:
                    obj_type = 1  # Circle
                elif type_field & 2:
                    obj_type = 2  # Slider
                elif type_field & 8:
                    obj_type = 3  # Spinner
                if type_field & 4:
                    new_combo = 1
                hitsound = int(parts[4]) if len(parts) > 4 else 0
                hit_objs.append((time, x, y, obj_type, new_combo, hitsound))
        print(f"Parsed {osu_path}: {len(hit_objs)} hit objects, Types: {np.unique([o[3] for o in hit_objs])}")
    return hit_objs, od

def prepare_data():
    X_audio = []
    X_difficulty = []
    y_hit_objs = []
    y_obj_types = []
    max_time = 120000

    for mp3_file in os.listdir(mp3_dir):
        if not mp3_file.endswith(".mp3"):
            continue
        
        song_id = mp3_file.replace("song", "").replace(".mp3", "")
        mp3_path = os.path.join(mp3_dir, mp3_file)
        beat_times, energy, tempo = extract_audio(mp3_path)
        
        if beat_times is None:
            continue

        max_beat = max(beat_times) if max(beat_times) > 0 else 1.0
        max_energy = np.max(energy) if np.max(energy) > 0 else 1.0
        X = np.column_stack((
            beat_times / max_beat,
            energy / max_energy,
            np.full(len(beat_times), tempo / 240)
        ))
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print(f"Skipping {mp3_file}: Invalid X values")
            continue

        print(f"X shape for {mp3_file}: {X.shape}, Sample X: {X[:5]}")

        for osu_file in os.listdir(osu_dir):
            if osu_file.startswith(f"song{song_id}_OD"):
                osu_path = os.path.join(osu_dir, osu_file)
                hit_objs, od = parse_osu(osu_path)
                y = np.array(hit_objs, dtype=np.float32)
                y[:, 0] /= max_time
                y[:, 1] /= 512
                y[:, 2] /= 384
                y[:, 5] /= 255
                obj_types = y[:, 3].astype(int) - 1
                
                if np.any(obj_types < 0) or np.any(obj_types > 2):
                    print(f"Skipping {osu_file}: Invalid obj_types {np.unique(obj_types)}")
                    continue
                
                if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                    print(f"Skipping {osu_file}: Invalid y values")
                    continue

                min_len = min(len(X), len(y))
                if min_len < 10:
                    print(f"Skipping {osu_file}: Too few aligned objects ({min_len})")
                    continue

                X_audio.append(X[:min_len])
                X_difficulty.append(np.full(min_len, od / 10))
                y_hit_objs.append(y[:min_len, [0, 1, 2, 4, 5]])
                y_obj_types.append(obj_types[:min_len])
    
    return X_audio, X_difficulty, y_hit_objs, y_obj_types

if __name__ == "__main__":
    X_audio, X_difficulty, y_hit_objs, y_obj_types = prepare_data()
    print(f"Processed {len(X_audio)} beatmaps")