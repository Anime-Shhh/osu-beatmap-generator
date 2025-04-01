import os

base_path = "stableOsuData/"
extensions = [".mp3", ".wav", ".ogg"]


def count_audio_files():
    mp3_count = 0
    non_mp3_count = 0
    no_audio = 0
    total_folders = 0

    # traverse all songs
    for song in os.listdir(base_path):
        # make new path for new song in which u check for audio
        song_path = os.path.join(base_path, song)

        # ensure that it is a valid song/folder
        if os.path.isdir(song_path):
            total_folders += 1
            has_audio = False
            is_mp3 = False

            # traverse the files to find a mp3
            for file in os.listdir(song_path):
                # ext = file.split()[1].lower()
                """os's split is needed because it splits by 
                   name and . instead of a space"""
                ext = os.path.splitext(file)[1].lower()

                if ext in extensions:
                    has_audio = True
                    if ext == extensions[0]:
                        is_mp3 = True
                    break

            if has_audio:
                if is_mp3:
                    mp3_count += 1
                else:
                    non_mp3_count += 1
            else:
                no_audio += 1

    print(f"Total Songs: {total_folders}")
    print(f"Total mp3 files: {mp3_count}")
    print(f"Total non_mp3 audio files: {non_mp3_count}")
    print(f"Total songs with no audio files: {no_audio}")
    print(f"mp3s/songs  = {mp3_count}/{total_folders}")


if __name__ == "__main__":
    count_audio_files()
