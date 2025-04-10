import os
import numpy as np
import tensorflow as tf
import librosa
import shutil
import zipfile

model_path = "osu_generator_normalized.h5"
test_song_path = "test_songs/apothecary.mp3"
temp_dir = "temp_dir"
output_dir = "output"

def extract_audio_features(test_song_path):
    y, sr = librosa.load(test_song_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    energy = librosa.feature.rms(y=y)[0]
    print(f"Beat times (first 5): {beat_times[:5]}, Length: {len(beat_times)}")
    return beat_times, energy[:len(beat_times)]

def prepare_input(beat_times, energy, od, max_len=1007):
    max_beat = max(beat_times)
    X = np.column_stack((beat_times / max_beat, energy / np.max(energy)))
    X_padded = tf.keras.preprocessing.sequence.pad_sequences(
        [X], maxlen=max_len, padding='post', dtype='float32'
    )
    od_padded = tf.keras.preprocessing.sequence.pad_sequences(
        [np.full(len(X), od / 10)], maxlen=max_len, padding='post', dtype='float32'
    )
    return np.concatenate([X_padded, od_padded[..., np.newaxis]], axis=-1)

def generate_hit_objects(model, mp3_path, od=4.0):
    beat_times, energy = extract_audio_features(mp3_path)
    if len(beat_times) == 0:
        raise ValueError("No beats detected in the mp3")

    X = prepare_input(beat_times, energy, od)
    predictions = model.predict(X)
    predictions = predictions[0, :len(beat_times)]

    print(f"Raw predictions (first 5): {predictions[:5]}")
    
    hit_objects = []
    max_time = 120000  # Match training max_time
    for i, pred in enumerate(predictions):
        time, x, y, obj_type, new_combo, hitsound = pred
        time = int(max(0, time) * max_time)  # Denormalize
        x = int(np.clip(x * 512, 0, 512))
        y = int(np.clip(y * 384, 0, 384))
        obj_type = int(np.round(np.clip(obj_type, 1, 3)))
        new_combo = int(np.round(np.clip(new_combo, 0, 1)))
        hitsound = int(np.round(np.clip(hitsound, 0, 255)))

        type_field = (1 if obj_type == 1 else 2 if obj_type == 2 else 8) | (4 if new_combo else 0)
        hit_objects.append((time, x, y, type_field, hitsound))

    print(f"Processed hit objects (first 5): {hit_objects[:5]}")
    return hit_objects

def write_osu_file(hit_objects, mp3_filename, od, osu_file_path):
    header = (
        "osu file format v14\n\n"
        "[General]\n"
        f"AudioFilename: {mp3_filename}\n"
        "AudioLeadIn: 0\n"
        "PreviewTime: -1\n"
        "Countdown: 0\n"
        "SampleSet: Normal\n"
        "StackLeniency: 0.7\n"
        "Mode: 0\n"
        "LetterboxInBreaks: 0\n"
        "WidescreenStoryboard: 0\n\n"
        "[Metadata]\n"
        "Title:Generated Beatmap\n"
        "Artist:Unknown\n"
        "Creator:Anime-Shhh\n"
        f"Version:AI Generated OD{od:.1f}\n"
        "Source:\n"
        "Tags:AI osu\n"
        "BeatmapID:0\n"
        "BeatmapSetID:-1\n\n"
        "[Difficulty]\n"
        f"OverallDifficulty:{od}\n"
        "ApproachRate:5\n"
        "HPDrainRate:5\n"
        "CircleSize:4\n"
        "SliderMultiplier:1.4\n"
        "SliderTickRate:1\n\n"
        "[Events]\n"
        "0,0,\"background.jpg\",0,0\n\n"
        "[TimingPoints]\n"
        "0,500,4,2,0,100,1,0\n\n"
        "[HitObjects]\n"
    )
    
    with open(osu_file_path, 'w', encoding='utf-8') as f:
        f.write(header)
        for time, x, y, type_field, hitsound in hit_objects:
            f.write(f"{x},{y},{time},{type_field},{hitsound},\n")

def create_osz(mp3_path, od=4.0):
    model = tf.keras.models.load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    mp3_filename = os.path.basename(mp3_path)
    mp3_name = os.path.splitext(mp3_filename)[0]
    
    temp_mp3_path = os.path.join(temp_dir, mp3_filename)
    shutil.copy(mp3_path, temp_mp3_path)
    
    hit_objects = generate_hit_objects(model, mp3_path, od)
    osu_filename = f"{mp3_name}_OD{od:.1f}.osu"
    osu_file_path = os.path.join(temp_dir, osu_filename)
    write_osu_file(hit_objects, mp3_filename, od, osu_file_path)
    
    osz_filename = f"{mp3_name}_OD{od:.1f}.osz"
    osz_path = os.path.join(output_dir, osz_filename)
    
    with zipfile.ZipFile(osz_path, 'w', zipfile.ZIP_DEFLATED) as osz:
        osz.write(temp_mp3_path, mp3_filename)
        osz.write(osu_file_path, osu_filename)
    
    shutil.rmtree(temp_dir)
    print(f"Generated .osz saved as {osz_path}")

def main():
    create_osz(test_song_path)

if __name__ == "__main__":
    main()