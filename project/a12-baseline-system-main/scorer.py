"""
Back-end Scoring:
- cosine similarity
- + adaptive score normalization
- ++ Quality-aware score calibration Function
- ???

"""
import os
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from scipy import spatial
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model import resnet as model_2d
from model import tdnn as model_1d
from config.config_scoring import Config
from data_processor.dataset import WavDataset
from data_processor.processing import load_wav, mean_std_norm_1d, truncate_speech
from tool.utils import compute_eer


def score(EPOCH=43):
    opt = Config()
    if opt.onlyscoring:
        embd_dict = np.load(
            "EXP/%s/%s_%s.npy" % (opt.save_dir, opt.save_name, EPOCH),
            allow_pickle=True,
        ).item()
        eer, _, cost, _ = get_eer(
            embd_dict, trial_file="DATA/%s/trials" % opt.val_dir)
        print("Epoch %d\t  EER %.4f\t  cost %.4f\n" %
              (EPOCH, eer * 100, cost))

    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

        # validation dataset
        val_dataset = WavDataset(opt=opt, train_mode=False)
        val_dataloader = DataLoader(val_dataset, num_workers=opt.workers, batch_size=1, pin_memory=True)

        if opt.conv_type == "1D":
            model = getattr(model_1d, opt.model)(
                in_dim=opt.in_planes,
                embedding_size=opt.embd_dim,
                hidden_dim=opt.hidden_dim,
            ).cuda()  # tdnn, ecapa_tdnn
        elif opt.conv_type == "2D":
            model = getattr(model_2d, opt.model)(
                in_planes=opt.in_planes, embedding_size=opt.embd_dim
            ).cuda()  # resnet

        print("Load EXP/%s/model_%d.pkl" % (opt.save_dir, EPOCH))
        checkpoint = torch.load("EXP/%s/model_%d.pkl" %
                                (opt.save_dir, EPOCH))
        model.load_state_dict(checkpoint["model"])
        model = nn.DataParallel(model)
        eer, cost = validate(model, val_dataloader, EPOCH, opt)
        print("Epoch %d\t  EER %.4f\t  cost %.4f\n" %
              (EPOCH, eer * 100, cost))


def get_eer(embd_dict, trial_file):
    true_score = []
    false_score = []

    with open(trial_file) as fh:
        for line in fh:
            line = line.strip()
            utt1, utt2, key = line.split()
            result = 1 - spatial.distance.cosine(embd_dict[utt1], embd_dict[utt2])
            if key == "1":
                true_score.append(result)
            elif key == "0":
                false_score.append(result)
    eer, threshold, mindct, threashold_dct = compute_eer(
        np.array(true_score), np.array(false_score)
    )
    return eer, threshold, mindct, threashold_dct


def validate(model, val_dataloader, epoch, opt):
    model.eval()
    embd_dict = {}
    with torch.no_grad():
        for j, (feat, utt) in tqdm(enumerate(val_dataloader)):
            outputs = model(feat.cuda())
            for i in range(len(utt)):
                # print(j, utt[i], feat.shape[2])
                embd_dict[utt[i]] = outputs[i, :].cpu().numpy()
    np.save("EXP/%s/%s_%s.npy" %
            (opt.save_dir, opt.save_name, epoch), embd_dict)
    if opt.scoring:
        eer, _, cost, _ = get_eer(
            embd_dict, trial_file="DATA/%s/trials" % opt.val_dir)
    else:
        eer, cost = 1, 1

    return eer, cost


def compare(filename1, filename2, EPOCH=43):
    opt = Config()
    transforms = torchaudio.transforms.MelSpectrogram(
        sample_rate=opt.fs,
        n_fft=opt.nfft,
        win_length=int(opt.fs * opt.win_len),
        hop_length=int(opt.fs * opt.hop_len),
        n_mels=opt.n_mels,
    )

    if opt.conv_type == "1D":
        model = getattr(model_1d, opt.model)(
            in_dim=opt.in_planes,
            embedding_size=opt.embd_dim,
            hidden_dim=opt.hidden_dim,
        ).cuda()  # tdnn, ecapa_tdnn
        model.eval()
    elif opt.conv_type == "2D":
        model = getattr(model_2d, opt.model)(
            in_planes=opt.in_planes, embedding_size=opt.embd_dim
        ).cuda()  # resnet
        model.eval()

    checkpoint = torch.load("EXP/%s/model_%d.pkl" % (opt.save_dir, EPOCH))
    model.load_state_dict(checkpoint["model"])
    model = nn.DataParallel(model)

    signal1 = load_wav(filename1, opt.max_frames, opt.fs, False)
    feat1 = torch.log(transforms(mean_std_norm_1d(signal1)) + 1e-6)
    feat1 = feat1 - feat1.mean(axis=1).unsqueeze(1)
    print(feat1.shape)

    signal2 = load_wav(filename2, opt.max_frames, opt.fs, False)
    feat2 = torch.log(transforms(mean_std_norm_1d(signal2)) + 1e-6)
    feat2 = feat2 - feat2.mean(axis=1).unsqueeze(1)
    print(feat2.shape)

    feat1 = feat1.unsqueeze(0)
    feat2 = feat2.unsqueeze(0)
    output1 = model(feat1)
    output2 = model(feat2)

    return 1 - spatial.distance.cosine(output1[0].cpu().detach().numpy(), output2[0].cpu().detach().numpy())


if __name__ == "__main__":
    score(EPOCH=43)
