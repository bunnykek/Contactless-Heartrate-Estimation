"""Trainer for DeepPhys."""

import os
import json
import pickle
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from model import *
from tqdm import tqdm


MODEL = {
    'Cnn_Lstm': Cnn_Lstm,
    'Mobilenet_Lstm': Mobilenet_Lstm,
    'CNN_biLSTM': CNN_biLSTM,
    'CNN_gru': CNN_gru,
    'Fft_Cnn_Lstm': Fft_Cnn_Lstm,
    'Resnet101_Lstm': Resnet101_Lstm,
    'Resnet50_Lstm': Resnet50_Lstm,
    'DeepPhys': DeepPhys
}


class Trainer():

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        print("model_dir", self.model_dir)
        print("model_file_name", self.model_file_name)
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.epoch = 0

        self.model = MODEL[config.MODEL.NAME](img_size=config.TRAIN.DATA.PREPROCESS.RESIZE.H).to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.criterion = torch.nn.MSELoss()
            self.optimizer = optim.AdamW(self.model.parameters(), lr=config.TRAIN.LR)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        
        mean_training_losses = []
        mean_valid_losses = []
        lrs = []

        json_output_dir = os.path.join(self.model_dir, 'stats_json')
        if not os.path.exists(json_output_dir):
            os.makedirs(json_output_dir)

        json_list = os.listdir(json_output_dir)
        if len(json_list)>0:
            with open(os.path.join(json_output_dir, json_list[-1])) as f:
                prev_json = json.load(f)
            mean_training_losses = prev_json['mean_training_losses']
            mean_valid_losses = prev_json['mean_valid_losses']
            lrs = prev_json['lrs']

        pth_output_dir = os.path.join(self.model_dir, "pth_files")
        if not os.path.exists(pth_output_dir):
            os.makedirs(pth_output_dir)

        pth_files = os.listdir(pth_output_dir)
        
        if len(pth_files)>0:
            self.load_checkpoint(os.path.join(pth_output_dir, pth_files[-1]))
            print("checkpoint epoch:", self.epoch)
        else:
            self.epoch=-1
            print("Fresh train")

        for epoch in range(self.epoch+1, self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                # print(np.array(batch[0]).shape) #(2, 180, 6, 72, 72)
                # print(np.array(batch[1]).shape) #(2, 180)
                # print(np.array(batch[2]).shape) #(2,)
                # print(np.array(batch[3]).shape) #(2,)
                # print(len(batch))
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].to(self.device), batch[1].to(self.device)
                N, D, C, H, W = data.shape   #(2, 180, 6, 72, 72)
                data = data.view(N * D, C, H, W)   #(2, 180, 6, 72, 72) -> (360, 6, 72, 72)
                labels = labels.view(-1, 1)  #(2, 180) -> (360, 1)
                # print(labels.shape)
                # print(labels)
                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                # print(pred_ppg.shape)
                # print(labels.shape)
                loss = self.criterion(pred_ppg, labels)
                loss.backward()

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                tbar.set_postfix({"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"]})

            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(train_loss))

            self.save_model(epoch)
            pth = os.listdir(pth_output_dir)
            if len(pth)==6:
                os.remove(os.path.join(pth_output_dir, pth[0]))

            valid_loss = self.valid(data_loader)
            mean_valid_losses.append(valid_loss)
            print('validation loss: ', valid_loss)
            
            with open(os.path.join(json_output_dir, f"epoch_{str(epoch).zfill(2)}.json"), 'w+') as f:   
                json.dump({
                    'mean_training_losses': mean_training_losses,
                    'mean_valid_losses': mean_valid_losses,
                    'lrs': lrs
                }, f, indent=4)
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)
            
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(
                    self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                data_valid = data_valid.view(N * D, C, H, W)
                labels_valid = labels_valid.view(-1, 1)
                pred_ppg_valid = self.model(data_valid)
                loss = self.criterion(pred_ppg_valid, labels_valid)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        config = self.config
        
        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            
            self.load_checkpoint(self.config.INFERENCE.MODEL_PATH, test=True)
            print("Testing uses pretrained model!")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, "pth_files", self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1).zfill(2) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.load_checkpoint(last_epoch_model_path, test=True)
            else:
                best_model_path = os.path.join(
                    self.model_dir, "pth_files", self.model_file_name + '_Epoch' + str(self.best_epoch).zfill(2) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.load_checkpoint(best_model_path, test=True)

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        with torch.no_grad():
            for _, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                pred_ppg_test = self.model(data_test)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    labels_test = labels_test.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
        
        print('')
        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        """Inits parameters from args and the writer for TensorboardX."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, "pth_files", self.model_file_name + '_Epoch' + str(index).zfill(2) + '.pth')
        state = {
                'epoch': index,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()
                }
        torch.save(state, model_path)

    def load_checkpoint(self, filepath, test=False):
        checkpoint = torch.load(filepath)
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        if not test:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

    def save_test_outputs(self, predictions, labels, config):
    
        output_dir = os.path.join(self.model_dir, 'test_results', "saved_test_outputs")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Filename ID to be used in any output files that get saved
        if config.TOOLBOX_MODE == 'train_and_test':
            filename_id = config.MODEL.NAME
        elif config.TOOLBOX_MODE == 'only_test':
            model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
            filename_id = config.MODEL.NAME
        else:
            raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')
        output_path = os.path.join(output_dir, filename_id + '_outputs.pickle')

        data = dict()
        data['predictions'] = predictions
        data['labels'] = labels
        data['label_type'] = config.TEST.DATA.PREPROCESS.LABEL_TYPE
        data['fs'] = config.TEST.DATA.FS

        with open(output_path, 'wb') as handle: # save out frame dict pickle file
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Saving outputs to:', output_path)

    def plot_losses_and_lrs(self, train_loss, valid_loss, lrs, config):

        output_dir = os.path.join(config.MODEL.MODEL_DIR, 'loss_plots')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Filename ID to be used in plots that get saved
        if config.TOOLBOX_MODE == 'train_and_test':
            filename_id = self.model_file_name
        else:
            raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')
        
        # Create a single plot for training and validation losses
        plt.figure(figsize=(10, 6))
        epochs = range(0, len(train_loss))  # Integer values for x-axis
        plt.plot(epochs, train_loss, label='Training Loss')
        plt.plot(epochs, valid_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{filename_id} Losses')
        plt.legend()
        plt.xticks(epochs)

        # Set y-axis ticks with more granularity
        ax = plt.gca()
        ax.yaxis.set_major_locator(MaxNLocator(integer=False, prune='both'))

        loss_plot_filename = os.path.join(output_dir, filename_id + '_losses.pdf')
        plt.savefig(loss_plot_filename, dpi=300)
        plt.close()

        # Create a separate plot for learning rates
        plt.figure(figsize=(6, 4))
        scheduler_steps = range(0, len(lrs))
        plt.plot(scheduler_steps, lrs, label='Learning Rate')
        plt.xlabel('Scheduler Step')
        plt.ylabel('Learning Rate')
        plt.title(f'{filename_id} LR Schedule')
        plt.legend()

        # Set y-axis values in scientific notation
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))  # Force scientific notation

        lr_plot_filename = os.path.join(output_dir, filename_id + '_learning_rates.pdf')
        plt.savefig(lr_plot_filename, bbox_inches='tight', dpi=300)
        plt.close()

        print('Saving plots of losses and learning rates to:', output_dir)
