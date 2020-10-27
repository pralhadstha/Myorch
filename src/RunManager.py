import torch
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import time
import json
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import itertools
from collections import OrderedDict
from EarlyStopping import EarlyStopping

class RunManager(object):
    def __init__(self, utils = None):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.epoch_validation_loss = 0
        self.epoch_validation_num_correct = 0
        self.epoch_validation_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.network_train = None
        self.loader = None
        self.validation_loader = None
        self.tb = None
        self.estopping = None
        self.estop = None
        self.optimizer = None

        self.mode_test = False

        self.actuals_train = []
        self.actuals_validation = []
        self.predictions_train = []
        self.predictions_validation = []
        self.actuals_test = []
        self.predictions_test = []

        # Contains the actual value that was predicted by the model.
        self.predictions_original_train = []
        self.predictions_original_validation = []
        self.predictions_original_test = []

        self.target_names = ['Alive', 'Dead']

        self.utils = utils
        self.device = 'cpu' # Because all the calculation are done in cpu and the y, yhat should also be in cpu
        self.flush_tb_epoch = 5 # Write the tb result to disk such that we won't get stuck with nothing in the tensorboard result when the system crashes.

    def begin_run(self, run, network, loader, optimizer):
        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader
        self.optimizer = optimizer
        self.device = self.network.device # or, utils.get_param('device')
        if (self.utils == None):
            self.tb = SummaryWriter()
            return
        self.tb = self.utils.custom_summary_writer(has_date_folder=True)
        if (not self.mode_test):
            self.estopping = EarlyStopping(patience = self.utils.get_param('patience'), verbose = self.utils.get_param('verbose'), utils = self.utils)

    def end_run(self, train_run = True):
        self.tb.close()
        self.epoch_count = 0
        if (train_run):
            self.utils.remove_intermediate_summary_writer()

    def begin_test_run(self, run, network, loader, optimizer=None):
        self.mode_test = True
        self.begin_run(run, network, loader, optimizer)
        self.begin_test()

    def end_test_run(self):
        self.end_test()
        self.end_run(False)

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.actuals_train = []
        self.actuals_validation = []
        self.predictions_train = []
        self.predictions_validation = []

        self.predictions_original_train = []
        self.predictions_original_validation = []


    def begin_validation_epoch(self, loader):
        self.validation_loader = loader
        self.epoch_validation_start_time = time.time()
        self.epoch_validation_loss = 0
        self.epoch_validation_num_correct = 0
        self.network_train = self.network

    def begin_test(self):
        self.epoch_start_time = time.time()
        
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.actuals_test = []
        self.predictions_test = []
        self.predictions_original_test = []

    def track_loss(self, loss, batch):
        self.epoch_loss += loss.item() * batch[0].shape[0]

    def track_num_correct(self, preds, actuals):
        # preds = self.utils.torch.sigmoid(preds)
        preds_pr = preds.detach().clone()
        if ((len(preds.shape) > 1)):
            if (preds.shape[1] > 1):
                preds = self.get_predicted_value(preds, 1)
                preds_pr = self.get_correct_predicted_value(preds_pr, 1)
        if ((len(actuals.shape) > 1)):
            if (actuals.shape[1] > 1):
                actuals = self.get_predicted_value(actuals, 1)
        self.epoch_num_correct += self.get_num_correct(preds, actuals)
        self.actuals_train.append(actuals)
        self.predictions_train.append(preds)
        self.predictions_original_train.append(preds_pr)

    def track_validation_loss(self, loss, batch):
        self.epoch_validation_loss += loss.item() * batch[0].shape[0]

    def track_validation_num_correct(self, preds, actuals):
        preds_pr = preds.clone().detach()
        if ((len(preds.shape) > 1)):
            if (preds.shape[1] > 1):
                preds = self.get_predicted_value(preds, 1)
                preds_pr = self.get_correct_predicted_value(preds_pr, 1)
        if ((len(actuals.shape) > 1)):
            if (actuals.shape[1] > 1):
                actuals = self.get_predicted_value(actuals, 1)
        self.epoch_validation_num_correct += self.get_num_correct(preds, actuals)
        self.actuals_validation.append(actuals)
        self.predictions_validation.append(preds)
        self.predictions_original_validation.append(preds_pr)

    def track_test_num_correct(self, preds, actuals):
        preds_pr = preds.clone().detach()
        if ((len(preds.shape) > 1)):
            if (preds.shape[1] > 1):
                preds = self.get_predicted_value(preds, 1)
                preds_pr = self.get_correct_predicted_value(preds_pr, 1)
        if ((len(actuals.shape) > 1)):
            if (actuals.shape[1] > 1):
                actuals = self.get_predicted_value(actuals, 1)
        self.epoch_num_correct += self.get_num_correct(preds, actuals)
        self.actuals_test.append(actuals)
        self.predictions_test.append(preds)
        self.predictions_original_test.append(preds_pr)

    def get_num_correct(self, preds, actuals):
        return preds.eq(actuals).sum().item()

    def get_predicted_value(self, preds, dim = None, use_threshold=False):
        if (use_threshold):
            return self.threshold_(preds)
        return self.argmax_(preds, dim)

    def get_correct_predicted_value(self, preds, dim = None):
        return self.max_(preds, dim).values

    def argmax_(self, preds, dim = None):
        return preds.argmax(dim = dim).to(self.device)

    def max_(self, preds, dim = None):
        return preds.max(dim = dim)

    def threshold_(self, preds, threshold=0.5):
        return torch.where(preds > threshold, torch.ceil_(preds), torch.floor_(preds))

    def select_tensor_index(self, input, index_tensor, dim = 0, output = None):
        return torch.index_select(input, dim, index_tensor, output)

    def list_to_tensor_list(self, list, tensor_list=True, dim=0):
        if (tensor_list):
            return torch.cat(list, dim).cpu()

        return np.asarray(torch.cat(list, dim).cpu())

    def array_value_count(self, array, value = 0):
        return np.count_nonzero(array == value)

    def value_cleaner(self, input, actuals, indices = [0], index_dim = 0, unsqueeze_dim = 0, argmax_dim=None):
        group_size = 3
        input = input.view(input.shape[1]//group_size, group_size, -1)
        actuals = actuals.view(actuals.shape[1]//group_size, group_size, -1)
        actuals = actuals[:, -1, :] # removes the second dimension
        if (self.utils.check_single_target(self.run_params.criterion)):
            actuals = self.argmax_(actuals, argmax_dim)#.unsqueeze(unsqueeze_dim)
        # Convert x sample to 2d for RBF as we cannot go further from 3d with timeseries data.
        if (self.utils.get_param('2dInput', False)):
            input = input.view(input.shape[0], -1)
        if (not self.utils.check_float_target(self.run_params.criterion)):
            actuals = actuals.long()

        return input.to(self.device), actuals.to(self.device)

    def activated_yhat(self, yhat, criterion):
        if (self.utils.check_activation(criterion)):
            yhat = self.utils.torch.sigmoid(yhat)

        return yhat

    def confusion_matrix_(self, preds, actuals, ravel=True):
        matrix = confusion_matrix(actuals, preds)
        if (ravel):
            return matrix.ravel()

        return matrix

    def create_statistics(self, dict):
        results = self.create_stats_dict(dict)
        if (not self.mode_test):
            results = self.create_hparams(results)
        self.run_data.append(results)

    def run_statistics(self):
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        decimal = self.utils.get_param('pandasDecimal', '{:,.10f}')
        #pd.options.display.float_format = decimal.format
        if (not self.mode_test):
            df['loss_reduced'] = df['loss_reduced'].apply(lambda x: decimal.format(x) if not isinstance(x, str) else x)

        clear_output(wait=True)
        display(df)
