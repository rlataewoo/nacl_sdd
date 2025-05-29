import os
import argparse
from common.dataset import ASVspoofDataset_eval

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tensorboardX import SummaryWriter

from common.utils import load_config, setup_seed, read_metadata, read_metadata_itw
from model import XLSR_Conformer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm.auto import tqdm

class Evaluator:
    def __init__(self, args, model, score_path):
        
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        self.model = model.to(self.accelerator.device)
        
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.model = self.accelerator.prepare(self.model)

        if self.accelerator.is_main_process:
            score_file_path = os.path.join(score_path, 'scores.txt')
            self.score_file = open(score_file_path, 'w')
            self.written_ids = set()
        else:
            self.score_file = None
            self.written_ids = None

    def create_dataloader(self, dataset, shuffle=False, collate_fn=None):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)
    
    def evaluate(self, dataset):
        eval_dataloader = self.create_dataloader(dataset, collate_fn=dataset.collate_fn)
        eval_dataloader = self.accelerator.prepare(eval_dataloader)
        self.model.eval()
        with torch.no_grad():
            all_utt_ids = []
            all_scores = []

            for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
                inputs, lengths, utt_ids = batch
                inputs, lengths = inputs.to(self.accelerator.device), lengths.to(self.accelerator.device)

                outputs, _ = self.model(inputs, lengths)
                
                batch_score = outputs[:, 1].data.cpu().numpy().ravel()

                all_utt_ids.extend(utt_ids)
                all_scores.extend(batch_score)

            all_utt_ids = self.accelerator.gather_for_metrics(all_utt_ids)
            all_scores = self.accelerator.gather_for_metrics(all_scores)

            if self.accelerator.is_main_process:
                for f, cm in zip(all_utt_ids, all_scores):
                    if f not in self.written_ids:  # Check for duplicate IDs
                        self.score_file.write(f"{f} {cm}\n")
                        self.written_ids.add(f)  # Track the written ID
                self.score_file.flush()

    def close(self):
        if self.score_file:
            self.score_file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--variable', type=bool, default=False, help='Variable length input')
    args = parser.parse_args()

    # Set seed for reproducibility
    variable = args.variable
    args = load_config(args.config)
    args.model.variable = variable

    model = XLSR_Conformer(args.model)

    if args.training.average_model:
        if os.path.exists(os.path.join(args.training.log_dir, 'best_model_avg.pth')):
            model.load_state_dict(torch.load(os.path.join(args.training.log_dir, 'best_model_avg.pth')))
            print('Model loaded : {}'.format(os.path.join(args.training.log_dir, 'best_model_avg.pth')))
        else:
            model.load_state_dict(torch.load(os.path.join(args.training.log_dir, 'best_model_{}.pth'.format(0))))
            print('Model loaded : {}'.format(os.path.join(args.training.log_dir, 'best_model_{}.pth'.format(0))))
            sd = model.state_dict()
            for i in range(1, args.training.n_mejores):
                model.load_state_dict(torch.load(os.path.join(args.training.log_dir, 'best_model_{}.pth'.format(i))))
                print('Model loaded : {}'.format(os.path.join(args.training.log_dir, 'best_model_{}.pth'.format(i))))
                sd2 = model.state_dict()
                for key in sd.keys():
                    sd[key] = (sd[key]+sd2[key])
            for key in sd.keys():
                sd[key] = (sd[key]) / args.training.n_mejores
            model.load_state_dict(sd)
            print("Model loaded average of {} best models in {}".format(args.training.n_mejores, args.training.log_dir))
    else:
        model.load_state_dict(torch.load(os.path.join(args.training.log_dir, 'best_model.pth')))
        print('Model loaded : {}'.format(os.path.join(args.training.log_dir, 'best_model.pth')))      
    
    if variable:
        score_path = os.path.join(args.training.log_dir, 'Scores_VL_ITW')
    else:
        score_path = os.path.join(args.training.log_dir, 'Scores_ITW')
    
    if not os.path.exists(score_path):
        os.makedirs(score_path, exist_ok=True)

    file_eval = read_metadata_itw(dir_meta = "datasets/protocols/in_the_wild.eval.txt")
    print('no. of eval trials',len(file_eval))
    eval_set = ASVspoofDataset_eval(
        list_IDs = file_eval, 
        base_dir = "datasets/in_the_wild",
        variable = variable
    )

    evaluator = Evaluator(args.training, model, score_path)
    evaluator.evaluate(eval_set)
    evaluator.close()
    print('Scores written to ' + score_path + "/scores.txt")

if __name__ == '__main__':
    main()
