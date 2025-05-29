#!/usr/bin/env python

import sys, os.path
import numpy as np
import pandas as pd
import eval_metrics as em
from glob import glob

if len(sys.argv) != 3:
    print("CHECK: invalid input arguments. Please read the instruction below:")
    print(__doc__)
    exit(1)

submit_file = sys.argv[1]
cm_key_file = sys.argv[2]

def eval_to_score_file(score_file, cm_key_file):
    cm_data = pd.read_csv(cm_key_file, sep=',', engine='python', header=None, names=['file', 'speaker', 'label'])
    submission_scores = pd.read_csv(score_file, sep=' ', header=None, names=['file', 'score'], skipinitialspace=True)

    # print("cm_data first rows:\n", cm_data.head())
    # print("submission_scores first rows:\n", submission_scores.head())

    if len(submission_scores.columns) > 2:
        print(f'CHECK: submission has more columns ({len(submission_scores.columns)}) than expected (2). Check for leading/ending blank spaces.')
        exit(1)

    cm_scores = submission_scores.merge(cm_data, on='file', how='inner')

    bona_cm = cm_scores[cm_scores['label'] == 'bona-fide']['score'].values
    spoof_cm = cm_scores[cm_scores['label'] == 'spoof']['score'].values
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
    out_data = "eer: %.3f\n" % (100*eer_cm)
    print(out_data)
    return eer_cm

if __name__ == "__main__":

    if not os.path.isfile(submit_file):
        print("%s doesn't exist" % (submit_file))
        exit(1)
        
    if not os.path.isfile(cm_key_file):
        print("%s doesn't exist" % (cm_key_file))
        exit(1)

    _ = eval_to_score_file(submit_file, cm_key_file)