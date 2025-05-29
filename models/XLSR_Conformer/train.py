import os
import shutil
import argparse
from common.dataset import ASVspoofDataset

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tensorboardX import SummaryWriter

from common.utils import load_config, setup_seed, read_metadata
from model import XLSR_Conformer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from tqdm.auto import tqdm

import numpy as np

class Trainer:
    def __init__(self, args, model):

        self.log_dir = args.log_dir
       
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        print(f"Using mixed precision: {self.accelerator.mixed_precision}")
        
        if self.accelerator.is_main_process:
            self.writer = SummaryWriter(self.log_dir)
            self.writer_text = open(os.path.join(self.log_dir, "log.txt"), "w")
        
        self.model = model.to(self.accelerator.device)
        
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.num_workers = args.num_workers

        if args.optimizer == "adam":
            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif args.optimizer == "adamw":
            self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Invalid optimizer: {args.optimizer}")
       
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        weight = torch.FloatTensor([0.1, 0.9]).to(self.accelerator.device)
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def create_dataloader(self, dataset, shuffle=False, collate_fn=None):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, collate_fn=collate_fn)

    def train_epoch(self, train_dataloader):
        self.model.train()
        total_loss = []
        for batch in tqdm(train_dataloader, desc="Training", leave=False, dynamic_ncols=True):
            with torch.set_grad_enabled(True):
                self.optimizer.zero_grad()
                inputs, labels, lengths, utt_ids = batch

                outputs, _ = self.model(inputs, lengths)
                labels = labels.view(-1).to(outputs.device)
                loss = self.criterion(outputs, labels)
                
                self.accelerator.backward(loss)
                self.optimizer.step()
                total_loss.append(loss.item())

        return total_loss

    def validate(self, val_dataloader):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation", leave=False, dynamic_ncols=True):
                inputs, labels, lengths, utt_ids = batch
                
                outputs, _ = self.model(inputs, lengths)
                labels = labels.view(-1).to(outputs.device)
                loss = self.criterion(outputs, labels)
                total_loss.append(loss.item())

        return total_loss
    
    def save_model(self, epoch, best=False):
        model_to_save = self.accelerator.unwrap_model(self.model)
        model_path = os.path.join(self.log_dir, f"epoch_{epoch}.pth")
        # torch.save(model_to_save.state_dict(), model_path)
        # print(f"Model saved to {model_path}")

        if best:
            best_model_path = os.path.join(self.log_dir, "best_model.pth")
            torch.save(model_to_save.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")

    def train(self, train_dataset, val_dataset, args):
        train_dataloader = self.create_dataloader(train_dataset, shuffle=True, collate_fn=train_dataset.collate_fn)
        val_dataloader = self.create_dataloader(val_dataset, shuffle=False, collate_fn=val_dataset.collate_fn)
        train_dataloader, val_dataloader = self.accelerator.prepare(train_dataloader, val_dataloader)
        epochs = args.epochs
        
        patience = args.patience # 7 epochs
        not_improving = 0 
        n_mejores = args.n_mejores # 5

        bests = np.ones(n_mejores,dtype=float)*float('inf')
        best_loss = float('inf')

        for i in range(n_mejores):
            torch.save(self.accelerator.unwrap_model(self.model).state_dict(), os.path.join(self.log_dir, f"best_model_{i}.pth"))
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_dataloader)
            gathered_train_loss = self.accelerator.gather(torch.tensor(train_loss).to(self.accelerator.device))
            gathered_train_loss = gathered_train_loss.mean().item()
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {gathered_train_loss:.4f}")
            val_loss = self.validate(val_dataloader)
            gathered_val_loss = self.accelerator.gather(torch.tensor(val_loss).to(self.accelerator.device))
            gathered_val_loss = gathered_val_loss.mean().item()
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch + 1}/{epochs} - Validation Loss: {gathered_val_loss:.4f}")
                self.writer.add_scalar("Loss/train", gathered_train_loss, epoch)
                self.writer.add_scalar("Loss/val", gathered_val_loss, epoch)
                self.writer.flush()
                self.writer_text.write(f"Epoch {epoch + 1}/{epochs} - Train Loss: {gathered_train_loss:.4f}\n")
                self.writer_text.write(f"Epoch {epoch + 1}/{epochs} - Validation Loss: {gathered_val_loss:.4f}\n")
                self.writer_text.flush()                

                if best_loss > gathered_val_loss:
                    best_loss = gathered_val_loss
                    self.save_model(epoch + 1, best=True)
                    not_improving = 0
                else:
                    not_improving += 1

                for i in range(n_mejores):
                    if bests[i] > gathered_val_loss:
                        for t in range(n_mejores-1, i, -1):
                            bests[t] = bests[t-1]
                            os.system('mv {}/best_model_{}.pth {}/best_model_{}.pth'.format(self.log_dir, t-1, self.log_dir, t))
                        bests[i] = gathered_val_loss
                        torch.save(self.accelerator.unwrap_model(self.model).state_dict(), os.path.join(self.log_dir, f"best_model_{i}.pth"))
                        break
                
                print(f"Bests: {bests}")

                if not_improving >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    self.accelerator.wait_for_everyone()
                    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--seed', type=int, default=1234, help='Seed for reproducibility')
    parser.add_argument('--variable', type=bool, default=False, help='Training with variable length input')
    args = parser.parse_args()

    # Set seed for reproducibility
    setup_seed(args.seed)
    config_path = args.config
    variable = args.variable
    args = load_config(args.config)
    args.model.variable = variable

    if variable:
        args.training.log_dir = args.training.log_dir + '_VL'

    if not os.path.exists(args.training.log_dir):
        os.makedirs(args.training.log_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(args.training.log_dir, 'config.yaml'))

    model = XLSR_Conformer(args.model)
    print(args.model.name)

    labels_trn, files_id_train = read_metadata(dir_meta = args.data.path_train_protocol, is_eval=False)
    labels_dev, files_id_dev = read_metadata(dir_meta = args.data.path_val_protocol, is_eval=False)

    print('no. of training trials',len(files_id_train))
    print('no. of validation trials',len(files_id_dev))
    
    train_dataset = ASVspoofDataset(
        args = args.augmentation, 
        list_IDs = files_id_train, 
        labels = labels_trn,
        base_dir = args.data.path_train,
        variable = variable
    )

    val_dataset = ASVspoofDataset(
        args = args.augmentation, 
        list_IDs = files_id_dev, 
        labels = labels_dev,
        base_dir = args.data.path_val,
        variable = variable
    )

    trainer = Trainer(args.training, model)
    trainer.train(train_dataset, val_dataset, args.training)
    print("Training finished")

if __name__ == '__main__':
    main()
