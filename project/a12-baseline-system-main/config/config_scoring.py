class Config(object):
    save_dir = "fwwb_80FBANK_ECAPATDNN_AAMsoftmax_256_dist"
    val_dir = "fwwb_dev"
    save_name = "test"
    scoring = True  # True: extract and scoring
    # False: extract, not scoring
    onlyscoring = False  # True : has npy
    # False : no npy

    workers = 4
    batch_size = 1
    max_frames = 300

    fs = 16000
    nfft = 512
    win_len = 0.025
    hop_len = 0.01
    n_mels = 80

    conv_type = "1D"  # 1D, 2D
    model = "ECAPA_TDNN"  # ResNet34StatsPool,TDNN,ECAPA_TDNN
    in_planes = 80  # conv_type:1D, in_planes=n_mels; 2D, in_planes=32, 64
    embd_dim = 256
    hidden_dim = 1024  # ECAPA_TDNN

    gpu = "0"
    utt2wav_val = [line.split() for line in open("DATA/%s/wav.scp" % val_dir)]