import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F

from common.RawBoost import  process_Rawboost_feature

class ASVspoofDataset(Dataset):
    def __init__(self, args, list_IDs, labels, labels_mos, base_dir, threshold=3.5840, variable=False):
        self.args = args
        self.list_IDs = list_IDs
        self.labels = labels
        self.labels_mos = labels_mos
        self.base_dir = base_dir
        self.variable = variable
        if not self.variable:
            self.cut = 64600 # take ~4 sec audio (64600 samples)
        
        self.threshold = threshold # 3.5840

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        audio_path = os.path.join(self.base_dir, 'wav', utt_id + '.wav')
        X, fs = librosa.load(audio_path, sr=16000)
        if self.args is not None:
            X = process_Rawboost_feature(X, fs, self.args)
        
        if not self.variable:
            X = self.pad(X, self.cut)
            
        Y = self.labels[utt_id]
        if self.labels_mos is not None:
            Y_mos = self.labels_mos[utt_id]
            # Calculate Temperature based on Y (0 for fake, 1 for real)
            if Y == 0:  # Fake
                Temperature = 1 + (5.0 - self.threshold) / (self.threshold - 1.0) * (Y_mos - self.threshold) / 4.0
            else:  # Real
                Temperature = 1 - (self.threshold - 1.0) / (5.0 - self.threshold) * (Y_mos - self.threshold) / 4.0
        else:
            Temperature = 1.0
        
        return torch.FloatTensor(X), torch.LongTensor([Y]), torch.FloatTensor([Temperature]), len(X), utt_id

    def __len__(self):
        return len(self.list_IDs)
    
    def collate_fn(self, samples):
        _X, Y, T, len_x, utt_ids = zip(*samples)

        max_len = max(len_x)
        X = torch.zeros(len(Y), max_len)
        
        for i in range(len(Y)):
            X[i, :len_x[i]] = _X[i]

        Y = torch.cat(Y)
        T = torch.cat(T)
        len_x = torch.LongTensor(len_x)

        return X, Y, T, len_x, utt_ids

    def pad(self, x, max_len):
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        num_repeats = int(max_len / x_len) + 1
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
        return padded_x

