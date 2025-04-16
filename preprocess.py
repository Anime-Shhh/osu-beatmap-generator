import librosa
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import re

def mp3_to_spectrogram(mp3_path, sr=22050, n_mels=128, hop_length=512, max_frames=10000):
    try:
        y, sr = librosa.load(mp3_path, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        if mel_spec_db.shape[1] > max_frames:
            mel_spec_db = mel_spec_db[:, :max_frames]
        else:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, max_frames - mel_spec_db.shape[1])), mode='constant')
        return mel_spec_db
    except Exception as e:
        print(f"Error processing {mp3_path}: {e}")
        return None

def parse_osu_file(osu_path, grid_size=(16, 12)):
    hit_objects = []
    try:
        with open(osu_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            hit_object_section = False
            for line in lines:
                if line.strip() == '[HitObjects]':
                    hit_object_section = True
                    continue
                if hit_object_section and line.strip():
                    parts = line.strip().split(',')
                    if len(parts) < 4:
                        continue
                    x, y, time = int(parts[0]), int(parts[1]), int(parts[2])
                    hit_type = int(parts[3])
                    grid_x = min(int(x / 512 * grid_size[0]), grid_size[0] - 1)
                    grid_y = min(int(y / 384 * grid_size[1]), grid_size[1] - 1)
                    type_id = 0 if hit_type & 1 else 1 if hit_type & 2 else 2
                    hit_objects.append({'time': time, 'type': type_id, 'grid_x': grid_x, 'grid_y': grid_y})
    except Exception as e:
        print(f"Error parsing {osu RIFF (WAVE) fmt 
            return hit_objects

def preprocess_audio(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for mp3_file in os.listdir(input_dir):
        if mp3_file.endswith('.mp3'):
            mp3_path = os.path.join(input_dir, mp3_file)
            mel_spec = mp3_to_spectrogram(mp3_path)
            if mel_spec is not None:
                output_path = os.path.join(output_dir, mp3_file.replace('.mp3', '.npy'))
                np.save(output_path, mel_spec)
                print(f"Processed {mp3_file}")

def preprocess_beatmaps(mp3_dir, osu_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mp3_files = {f.replace('.mp3', '') for f in os.listdir(mp3_dir) if f.endswith('.mp3')}
    for osu_file in os.listdir(osu_dir):
        if osu_file.endswith('.osu'):
            # Extract base song name (e.g., 'song1' from 'song1_OD5.0.osu')
            match = re.match(r'^(.*?)(?:_OD\d+\.\d+)?\.osu$', osu_file)
            if not match:
                continue
            song_name = match.group(1)
            if song_name in mp3_files:
                osu_path = os.path.join(osu_dir, osu_file)
                hit_objects = parse_osu_file(osu_path)
                if hit_objects:
                    output_path = os.path.join(output_dir, osu_file.replace('.osu', '.npy'))
                    np.save(output_path, hit_objects)
                    print(f"Processed {osu_file}")

class OsuDataset(Dataset):
    def __init__(self, spec_dir, beatmap_dir, max_frames=10000):
        self.spec_files = [f for f in os.listdir(spec_dir) if f.endswith('.npy')]
        self.beatmap_files = [f for f in os.listdir(beatmap_dir) if f.endswith('.npy')]
        self.spec_dir = spec_dir
        self.beatmap_dir = beatmap_dir
        self.max_frames = max_frames

    def __len__(self):
        return len(self.beatmap_files)

    def __getitem__(self, idx):
        beatmap_file = self.beatmap_files[idx]
        # Find corresponding spectrogram (e.g., 'song1_OD5.0.npy' -> 'song1.npy')
        song_name = re.match(r'^(.*?)(?:_OD\d+\.\d+)?\.npy$', beatmap_file).group(1)
        spec_file = f"{song_name}.npy"
        
        spec_path = os.path.join(self.spec_dir, spec_file)
        beatmap_path = os.path.join(self.beatmap_dir, beatmap_file)
        
        # Load spectrogram
        spec = np.load(spec_path)
        
        # Load and tokenize hit objects
        hit_objects = np.load(beatmap_path, allow_pickle=True)
        tokens = []
        for obj in hit_objects:
            time_bin = min(int(obj['time'] / 1000 * 22050 / 512), self.max_frames - 1)
            tokens.append([time_bin, obj['type'], obj['grid_x'], obj['grid_y']])
        
        max_tokens = 300
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        else:
            tokens = tokens + [[0, 0, 0, 0]] * (max_tokens - len(tokens))
        
        return torch.tensor(spec, dtype=torch.float32), torch.tensor(tokens, dtype=torch.long)

if __name__ == "__main__":
    # Preprocess audio
    preprocess_audio('data/organized/mp3', 'data/spectrograms')
    # Preprocess beatmaps
    preprocess_beatmaps('data/organized/mp3', 'data/organized/osu', 'data/parsed_beatmaps')
    # Test dataset
    dataset = OsuDataset('data/spectrograms', 'data/parsed_beatmaps')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for specs, tokens in dataloader:
        print(f"Sample batch: specs shape {specs.shape}, tokens shape {tokens.shape}")
        break