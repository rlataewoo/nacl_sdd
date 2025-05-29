import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F

from common.RawBoost import  process_Rawboost_feature

class ASVspoofDataset(Dataset):
    def __init__(self, args, list_IDs, labels, labels_mos, base_dir, difficulty=None, variable=False):
        self.args = args
        
        self.list_IDs = list_IDs
        self.labels = labels
        self.labels_mos = labels_mos

        self.base_dir = base_dir
        self.variable = variable
        if not self.variable:
            self.cut = 64600 # take ~4 sec audio (64600 samples)
        
        if difficulty is not None:
            self.list_IDs = self._get_subset(difficulty)

    def _get_subset(self, difficulty):
        subset = []
        for utt_id in self.list_IDs:
            Y = self.labels[utt_id]
            Y_mos = self.labels_mos[utt_id]
            Y_mos = (Y_mos - 1.0) / 4.0
            if Y == 1: # if bonafide
                H = 1 - Y_mos
            else: # if spoof
                H = Y_mos
            if H <= difficulty:
                subset.append(utt_id)
        return subset

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        audio_path = os.path.join(self.base_dir, 'wav', utt_id + '.wav')
        X, fs = librosa.load(audio_path, sr=16000)
        if self.args is not None:
            X = process_Rawboost_feature(X, fs, self.args)
        
        if not self.variable:
            X = self.pad(X, self.cut)
            
        Y = self.labels[utt_id]
        
        return torch.FloatTensor(X), torch.LongTensor([Y]), len(X), utt_id

    def __len__(self):
        return len(self.list_IDs)
    
    def collate_fn(self, samples):
        _X, Y, len_x, utt_ids = zip(*samples)

        max_len = max(len_x)
        X = torch.zeros(len(Y), max_len)
        
        for i in range(len(Y)):
            X[i, :len_x[i]] = _X[i]

        Y = torch.cat(Y)
        len_x = torch.LongTensor(len_x)

        return X, Y, len_x, utt_ids

    def pad(self, x, max_len):
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        num_repeats = int(max_len / x_len) + 1
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
        return padded_x
