"""
data preprocessing:
- WADA-SNR
- voice activity detection
- auto
- self-supervised cleansing
- data grouping
- 

online data augmentation (multi-style training):

- waveform
    - speaker augmentation (speed/tempo)
    - additive noise (babble, noise, music)
    - multiplicative noise (a.k.a. reverberation augmentation)
    - power/volume
    - *resample
    - *reencode
    - 

online feature extraction:

- FBank/MFCC/LFBE/PLP
- mean-normalization
- SpecAugment (note the sequence)
- 

dataset fusion:
- *multireader
- *speakerstew
- 

"""
