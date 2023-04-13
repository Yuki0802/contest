import os
import random

from tqdm.auto import tqdm

print(os.getcwd())

# Path to the directory containing the audio dataset.
PATH_TO_DATASET = "E:\\aidatatang_200zh/corpus"

# Path to all the output files
# based on this.py file if you are running with PyCharm
PATH_TO_OUTPUT = "../DATA"

# Extension of the audio files.
AUDIO_FORMAT = ".wav"

# After we split the full audio path
# the index indicating which part is the speaker label.
SPEAKER_LABEL_INDEX = -2

# Noise dataset
PATH_TO_NOISE = [
    "E:\\musan/noise",
    "E:\\musan/music",
    "E:\\rirs_noises/",
]
FOLDER_TO_NOISE_OUTPUT = ["MUSAN", "MUSAN", "RIR_Noise"]
NAME_TO_NOISE_OUTPUT = ["noise_wav_list", "music_wav_list", "rir_list"]


# fwwb_trian：wav.scp, utt2spk, spk2utt
# fwwb_eval：wav.scp, trials（正负样本对）
def get_wav_list(mode: str):
    foldername = f"fwwb_{mode}"

    wav_scp, utt2spk, spk2utt = [], [], {}
    spks, trials = [], []

    # Find all files in PATH_TO_DATASET with the extension of AUDIO_FORMAT/MODE/.
    for filepath, _, filenames in tqdm(os.walk(os.path.join(PATH_TO_DATASET, mode))):
        for filename in filenames:
            if filename.endswith(AUDIO_FORMAT) and not filename.startswith("."):
                path = os.path.join(filepath, filename)
                spk = path.split(os.sep)[SPEAKER_LABEL_INDEX]
                utt = filename.strip(AUDIO_FORMAT)

                spks.append(spk)
                wav_scp.append(f"{utt} {path}\n")
                utt2spk.append(f"{utt} {spk}\n")

                if spk not in spk2utt:
                    spk2utt[spk] = [utt]
                else:
                    spk2utt[spk].append(utt)

    write_list(foldername, "wav.scp", wav_scp)

    if mode == "train":
        write_list(foldername, "utt2spk", utt2spk)
        write_dict(foldername, "spk2utt", spk2utt)

    # FIXME: Is it reasonable?
    # elif mode == "dev":
    #     for _ in range(20000):
    #         trials.extend(get_random_trials(spks, spk2utt))
    #     write_list(foldername, "trials", trials)


def get_random_trials(spks, spk2utt):
    spk = random.choice(spks)
    utt1, utt2 = random.sample(spk2utt[spk], 2)
    positive = f"{utt1} {utt2} 1\n"

    spk1, spk2 = random.sample(spks, 2)
    utt1, utt2 = random.choice(spk2utt[spk1]), random.choice(spk2utt[spk2])
    negative = f"{utt1} {utt2} 0\n"

    return positive, negative


def write_list(foldername: str, filename: str, data):
    save_path = os.path.join(PATH_TO_OUTPUT, foldername, filename)
    with open(save_path, "w") as fout:
        fout.write("".join(data))


def write_dict(foldername: str, filename: str, data):
    save_path = os.path.join(PATH_TO_OUTPUT, foldername, filename)
    with open(save_path, "w") as fout:
        for key, val in data.items():
            spk, utt = key, " ".join(val)
            fout.write(f"{spk} {utt}\n")


def get_noise_list():
    for foldername, filename, path in zip(
            FOLDER_TO_NOISE_OUTPUT, NAME_TO_NOISE_OUTPUT, PATH_TO_NOISE
    ):
        all_files = [
            os.path.join(dirpath, filename) + "\n"
            for dirpath, _, files in tqdm(os.walk(path))
            for filename in files
            if filename.endswith(AUDIO_FORMAT)
        ]

        if not os.path.exists(os.path.join(PATH_TO_OUTPUT, foldername)):
            os.mkdir(os.path.join(PATH_TO_OUTPUT, foldername))

        write_list(foldername, filename, all_files)


if __name__ == "__main__":
    get_noise_list()

    for mode in ["train", "dev"]:
        folder = os.path.join(PATH_TO_OUTPUT, f"fwwb_{mode}")
        if not os.path.exists(folder):
            os.mkdir(folder)
        get_wav_list(mode)
