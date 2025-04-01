import os
import shutil

base_dir = "stableOsuData/Songs"
output_dir = "organized"
mp3_dir = os.path.join(output_dir, "mp3")
osu_dir = os.path.join(output_dir, "osu")

os.makedirs(mp3_dir, exist_ok=True)
os.makedirs(osu_dir, exist_ok=True)


def get_overall_difficulty(osu_file):
    with open(osu_file, 'r', encoding='utf=8') as f:
        in_difficulty_section = False
        for line in f:
            line = line.strip()
            if line == "[Difficulty]":
                in_difficulty_section = True
            elif in_difficulty_section and line.startswith("["):
                break
            elif in_difficulty_section and line.startswith("OverallDifficulty:"):
                return float(line.split(":")[1])
    return None

def process_song(song_path, song_id):
    #make sure song is a mp3 song
    mp3_file = None
    osu_files = []

    for file in os.listdir(song_path):
        file_path = os.path.join(song_path, file)
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".mp3":
            mp3_file = file_path
        elif ext == ".osu":
            osu_files.append(file_path)

    #only process song if there is an mp3
    if mp3_file:
        new_mp3_path = os.path.join(mp3_dir, f"song{song_id}.mp3")
        shutil.copy(mp3_file, new_mp3_path)

        for osu_file in osu_files:
            od = get_overall_difficulty(osu_file)

            if od is not None:
                new_osu_path = os.path.join(osu_dir, f"song{song_id}_OD{od}.osu")
                shutil.copy(osu_file, new_osu_path)
            else:
                print("No Overall Difficulty Found")

def main():
    song_id = 1
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            process_song(folder_path, song_id)
            song_id += 1

if __name__ == "__main__":
    main()
