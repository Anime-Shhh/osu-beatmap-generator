import librosa
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

def mp3_to_spectrogram(mp3_path, sr=22050, n_mels=128, hop_length=512, n_fft=1024, max_frames=10000, min_duration=5.0):
    try:
        y, sr = librosa.load(mp3_path, sr=sr, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < min_duration:
            logger.warning(f"Skipping {mp3_path}: Duration {duration}s is less than {min_duration}s")
            return None
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        if mel_spec_db.shape[1] > max_frames:
            mel_spec_db = mel_spec_db[:, :max_frames]
        else:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, max_frames - mel_spec_db.shape[1])), mode='constant')
        return mel_spec_db
    except Exception as e:
        logger.error(f"Failed to process {mp3_path}: {str(e)}")
        return None

def parse_osu_file(osu_path, grid_size=(16, 12)):
    try:
        hit_objects = []
        with open(osu_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            hit_object_section = False
            for line in lines:
                if line.strip() == '[HitObjects]':
                    hit_object_section = True
                    continue
                if hit_object_section and line.strip():
                    parts = line.strip().split(',')
                    x, y, time = int(parts[0]), int(parts[1]), int(parts[2])
                    hit_type = int(parts[3])
                    grid_x = min(int(x / 512 * grid_size[0]), grid_size[0] - 1)
                    grid_y = min(int(y / 384 * grid_size[1]), grid_size[1] - 1)
                    type_id = 0 if hit_type & 1 else 1 if hit_type & 2 else 2
                    hit_objects.append({'time': time, 'type': type_id, 'grid_x': grid_x, 'grid_y': grid_y})
        if not hit_objects:
            logger.warning(f"No hit objects found in {osu_path}")
            return None
        return hit_objects
    except Exception as e:
        logger.error(f"Failed to parse {osu_path}: {str(e)}")
        return None

def preprocess_audio(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for mp3_file in os.listdir(input_dir):
        if mp3_file.endswith('.mp3'):
            mp3_path = os.path.join(input_dir, mp3_file)
            output_path = os.path.join(output_dir, mp3_file.replace('.mp3', '.npy'))
            if os.path.exists(output_path):
                logger.debug(f"Overwriting existing spectrogram: {output_path}")
            mel_spec = mp3_to_spectrogram(mp3_path)
            if mel_spec is not None:
                np.save(output_path, mel_spec)
                logger.info(f"Processed {mp3_path} -> {output_path}")
            else:
                logger.warning(f"Skipped {mp3_path}")

def preprocess_beatmaps(mp3_dir, osu_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for mp3_file in os.listdir(mp3_dir):
        if mp3_file.endswith('.mp3'):
            song_name = mp3_file.replace('.mp3', '')
            osu_files = [f for f in os.listdir(osu_dir) if f.startswith(song_name) and f.endswith('.osu')]
            for osu_file in osu_files:
                osu_path = os.path.join(osu_dir, osu_file)
                output_path = os.path.join(output_dir, osu_file.replace('.osu', '.npy'))
                if os.path.exists(output_path):
                    logger.debug(f"Overwriting existing beatmap: {output_path}")
                hit_objects = parse_osu_file(osu_path)
                if hit_objects is not None:
                    np.save(output_path, hit_objects)
                    logger.info(f"Processed {osu_path} -> {output_path}")
                else:
                    logger.warning(f"Skipped {osu_path}")

class OsuDataset(Dataset):
    def __init__(self, spec_dir, beatmap_dir, max_frames=10000, max_tokens=1000):
        self.beatmap_files = [f for f in os.listdir(beatmap_dir) if f.endswith('.npy')]
        self.spec_dir = spec_dir
        self.beatmap_dir = beatmap_dir
        self.max_frames = max_frames
        self.max_tokens = max_tokens
        self.valid_pairs = []
        for beatmap_file in self.beatmap_files:
            song_name = '_'.join(beatmap_file.split('_')[:-1])
            spec_path = os.path.join(self.spec_dir, f"{song_name}.npy")
            if os.path.exists(spec_path):
                self.valid_pairs.append((spec_path, os.path.join(self.beatmap_dir, beatmap_file)))
            else:
                logger.warning(f"No spectrogram found for {beatmap_file}")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        spec_path, beatmap_path = self.valid_pairs[idx]
        spec = np.load(spec_path)
        hit_objects = np.load(beatmap_path, allow_pickle=True)
        tokens = []
        for obj in hit_objects:
            time_bin = min(int(obj['time'] / 1000 * 22050 / 512), self.max_tokens - 1)
            type_id = obj['type']
            grid_x = obj['grid_x']
            grid_y = obj['grid_y']
            if not (0 <= time_bin < self.max_tokens):
                logger.error(f"Invalid time_bin {time_bin} in {beatmap_path}, clamping to 0")
                time_bin = 0
            if not (0 <= type_id <= 2):
                logger.error(f"Invalid type_id {type_id} in {beatmap_path}, clamping to 0")
                type_id = 0
            if not (0 <= grid_x < 16):
                logger.error(f"Invalid grid_x {grid_x} in {beatmap_path}, clamping to 0")
                grid_x = 0
            if not (0 <= grid_y < 12):
                logger.error(f"Invalid grid_y {grid_y} in {beatmap_path}, clamping to 0")
                grid_y = 0
            tokens.append([time_bin, type_id, grid_x, grid_y])
        
        if len(tokens) > self.max_tokens:
            tokens = tokens[:self.max_tokens]
        else:
            tokens = tokens + [[0, 0, 0, 0]] * (self.max_tokens - len(tokens))
        
        logger.debug(f"Tokens for {beatmap_path}: shape {np.array(tokens).shape}, "
                     f"time_bin [{min(t[0] for t in tokens)}, {max(t[0] for t in tokens)}]")
        return torch.tensor(spec, dtype=torch.float32), torch.tensor(tokens, dtype=torch.long)

if __name__ == "__main__":
    preprocess_audio('data/organized/mp3', 'data/spectrograms')
    preprocess_beatmaps('data/organized/mp3', 'data/organized/osu', 'data/parsed_beatmaps')
    dataset = OsuDataset('data/spectrograms', 'data/parsed_beatmaps')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)