# Naturalness-Aware Curriculum Learning with Dynamic Temperature for Speech Deepfake Detection
This repository contains our implementation and pretrained models of the paper accepted by Interspeech 2025.
   * **Title**: Naturalness-Aware Curriculum Learning with Dynamic Temperature for Speech Deepfake Detection
   * **Author**: Taewoo Kim, Guisik Kim, Choongsang Cho, Young Han Lee
   * **Affiliation**: Korea Electronics Technology Institute (KETI)
   * [Paper](https://arxiv.org/abs/2505.13976)

## Environment
   * Docker: nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
   * Device: NVIDIA A40 48GB GPU

## Installation
1. Install pyenv
2. Clone this repository
3. Setup virtual environment and install python requirements.
```sh
pyenv install 3.8.0
pyenv virtualenv 3.8.0 nacl_sdd
pyenv local nacl_sdd
poetry env use python
poetry install
```
4. The Wav2vec2.0 model (XLS-R 300M) can be downloaded from [here](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/xlsr). 

## Data Preparation
1. Downlaod datasets
    * [ASVSpoof2019](https://www.asvspoof.org/index2019.html): Training and validation set
    * [ASVSpoof2021](https://www.asvspoof.org/index2021.html): Evaluation set 1
    * [InTheWild](https://deepfake-total.com/in_the_wild): Evaluation set 2

2. Download keys for evaluation
    * [LA keys](https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz)
    * [PA keys](https://www.asvspoof.org/asvspoof2021/PA-keys-full.tar.gz)
    * [DF keys](https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz)

3. Arrange datasets 
```
├── datasets
│   ├── ASVspoof2019_LA_train
│   │   └── wav/
│   ├── ASVspoof2019_LA_dev
│   │   └── wav/
│   ├── ASVspoof2021_LA_eval
│   │   └── wav/
│   ├── ASVspoof2021_DF_eval
│   │   └── wav/
│   ├── in_the_wild
│   │   └── wav/
│   │
│   ├── keys
│   │   └── DF/
│   │   └── LA/
│   │   └── PA/
│   │   └── ITW/
│   ├── protocols
│   ├── utmos
│
├── pretrained
├── results

```

## Pretrained Model and Results
Downlaod pretrained XLS-R Conformer models and results.
```sh
wget -O download.zip https://bit.ly/interspeech2025_NACL && unzip download.zip
```

## Usage
```sh
# Training
sh scripts/train.sh

# Evaluation for LA & DF
sh scripts/evaluate.sh

# Evaluation for ITW
sh scripts/evaluate_itw.sh
```

## Results
```sh
# Calculate Score for LA & DF
poetry run python common/eval/main.py --cm-score-file results/table_1/LA_fix/conformer_cl_dt.txt --track LA --subset eval
poetry run python common/eval/main.py --cm-score-file results/table_1/DF_fix/conformer_cl_dt.txt --track DF --subset eval

# Calculate Score for ITW (Table 3)
poetry run python common/eval/evaluate_in_the_wild.py results/table_3/ITW/conformer_cl_dt.txt datasets/keys/ITW/meta.csv
```

## Citation
```bibtex
@inproceedings{kim2025naturalness,
  author={Kim, Taewoo and Kim, Guisik and Cho, Choongsang and Lee, Young Han},
  booktitle={arXiv preprint arXiv:2505.13976},
  title={Naturalness-Aware Curriculum Learning with Dynamic Temperature for Speech Deepfake Detection},
  year={2025}
}
```
