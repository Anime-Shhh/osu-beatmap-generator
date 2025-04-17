import torch
import numpy as np
import os
import zipfile
import shutil
from preprocess import mp3_to_spectrogram
from model import TransformerDecoder
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_beatmap(model, mp3_path, output_osz_path, max_tokens=10000):
    model.eval()
    try:
        mel_spec = mp3_to_spectrogram(mp3_path)
        if mel_spec is None:
            logger.error(f"Failed to generate spectrogram for {mp3_path}")
            return
        
        spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).to(device)
        
        tokens = torch.zeros(1, max_tokens, 4, dtype=torch.long).to(device)
        
        with torch.no_grad():
            for i in range(max_tokens - 1):
                time_logits, type_logits, grid_x_logits, grid_y_logits = model(spec, tokens)
                tokens[0, i + 1, 0] = torch.argmax(time_logits[:, i], dim=-1)
                tokens[0, i + 1, 1] = torch.argmax(type_logits[:, i], dim=-1)
                tokens[0, i + 1, 2] = torch.argmax(grid_x_logits[:, i], dim=-1)
                tokens[0, i + 1, 3] = torch.argmax(grid_y_logits[:, i], dim=-1)
        
        hit_objects = []
        for token in tokens[0]:
            time_bin, type_id, grid_x, grid_y = token.tolist()
            if time_bin == 0:
                continue
            time_ms = int(time_bin * 512 / 22050 * 1000)
            x = int(grid_x * 512 / 16)
            y = int(grid_y * 384 / 12)
            hit_type = 1 if type_id == 0 else 2 if type_id == 1 else 8
            hit_objects.append(f'{x},{y},{time_ms},{hit_type},0,0:0:0:0:')
        
        temp_dir = 'temp_osz'
        os.makedirs(temp_dir, exist_ok=True)
        osu_filename = os.path.basename(mp3_path).replace('.mp3', '.osu')
        osu_path = os.path.join(temp_dir, osu_filename)
        
        with open(osu_path, 'w') as f:
            f.write('osu file format v14\n\n')
            f.write(f'[General]\nAudioFilename: {os.path.basename(mp3_path)}\n\n')
            f.write('[Metadata]\nTitle:Generated Beatmap\nArtist:Unknown\n\n')
            f.write('[Difficulty]\nOverallDifficulty:5\n\n')
            f.write('[Events]\n//Background and Video events\n\n')
            f.write('[TimingPoints]\n0,500,4,2,0,100,1,0\n\n')
            f.write('[HitObjects]\n')
            for obj in hit_objects:
                f.write(obj + '\n')
        
        with zipfile.ZipFile(output_osz_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(mp3_path, os.path.basename(mp3_path))
            zf.write(osu_path, osu_filename)
        
        shutil.rmtree(temp_dir)
        logger.info(f"Generated {output_osz_path}")
    except Exception as e:
        logger.error(f"Failed to generate beatmap for {mp3_path}: {str(e)}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerDecoder(max_tokens=10000).to(device)
    try:
        model.load_state_dict(torch.load('output/model.pth'))
        generate_beatmap(model, 'test_songs/apothecary.mp3', 'output/apothecary.osz')
    except FileNotFoundError as e:
        logger.error(f"Model or test song not found: {str(e)}")