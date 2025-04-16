import torch
import numpy as np

def generate_beatmap(model, mp3_path, output_osu_path, max_tokens=300):
    model.eval()
    # Process audio
    mel_spec = mp3_to_spectrogram(mp3_path)
    spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Initialize output sequence
    tokens = torch.zeros(1, max_tokens, 4, dtype=torch.long).to(device)
    
    with torch.no_grad():
        for i in range(max_tokens - 1):
            time_logits, type_logits, grid_x_logits, grid_y_logits = model(spec, tokens)
            tokens[0, i + 1, 0] = torch.argmax(time_logits[:, i], dim=-1)
            tokens[0, i + 1, 1] = torch.argmax(type_logits[:, i], dim=-1)
            tokens[0, i + 1, 2] = torch.argmax(grid_x_logits[:, i], dim=-1)
            tokens[0, i + 1, 3] = torch.argmax(grid_y_logits[:, i], dim=-1)
    
    # Convert tokens to .osu format
    hit_objects = []
    for token in tokens[0]:
        time_bin, type_id, grid_x, grid_y = token.tolist()
        if time_bin == 0:  # Skip padding
            continue
        # Convert back to time (ms)
        time_ms = int(time_bin * 512 / 22050 * 1000)
        # Convert grid to coordinates
        x = int(grid_x * 512 / 16)
        y = int(grid_y * 384 / 12)
        # Map type
        hit_type = 1 if type_id == 0 else 2 if type_id == 1 else 8
        hit_objects.append(f'{x},{y},{time_ms},{hit_type},0,0:0:0:0:')
    
    # Write .osu file
    with open(output_osu_path, 'w') as f:
        f.write('osu file format v14\n\n')
        f.write('[General]\nAudioFilename: input.mp3\n\n')
        f.write('[Metadata]\nTitle:Generated Beatmap\nArtist:Unknown\n\n')
        f.write('[Difficulty]\nOverallDifficulty:5\n\n')
        f.write('[Events]\n//Background and Video events\n\n')
        f.write('[TimingPoints]\n0,500,4,2,0,100,1,0\n\n')
        f.write('[HitObjects]\n')
        for obj in hit_objects:
            f.write(obj + '\n')

# Example usage
generate_beatmap(model, 'data/songs/test.mp3', 'data/output/test.osu')