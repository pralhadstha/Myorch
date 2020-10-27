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

    def end_epoch(self, hasValidation=False):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / (len(self.loader.dataset)/3)
        #accuracy = accuracy_score(self.actuals_train, self.predictions_train)
        roc_auc_train = roc_auc_score(self.list_to_tensor_list(self.actuals_train), self.list_to_tensor_list(self.predictions_train))

        if (hasValidation):
            epoch_validation_duration = time.time() - self.epoch_validation_start_time
            loss_validation = self.epoch_validation_loss / len(self.validation_loader.dataset)
            accuracy_validation = self.epoch_validation_num_correct / (len(self.validation_loader.dataset)/3)
            #accuracy_validation = accuracy_score(self.actuals_validation, self.predictions_validation)
            roc_auc_validation = roc_auc_score(self.list_to_tensor_list(self.actuals_validation), self.list_to_tensor_list(self.predictions_validation))

        if (hasValidation):
            self.tb.add_scalars('Loss', { 'train':loss, 'validation': loss_validation }, self.epoch_count)
            self.tb.add_scalars('Accuracy', { 'train':accuracy, 'validation': accuracy_validation }, self.epoch_count)

            self.tb.add_pr_curve('prec-rec-curve-train', self.list_to_tensor_list(self.actuals_train, False), self.list_to_tensor_list(self.predictions_original_train, False), self.epoch_count)
            self.tb.add_pr_curve('prec-rec-curve-val', self.list_to_tensor_list(self.actuals_validation, False), self.list_to_tensor_list(self.predictions_original_validation, False), self.epoch_count)
        else:
            self.tb.add_scalar('Loss', loss, self.epoch_count)
            self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)
            self.tb.add_pr_curve('prec-rec-curve-train', self.list_to_tensor_list(self.actuals_train, False), self.list_to_tensor_list(self.predictions_original_train, False), self.epoch_count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        if (hasValidation):
            for name, param in self.network_train.named_parameters():
                self.tb.add_histogram(f"{name}.train", param, self.epoch_count)
                self.tb.add_histogram(f'{name}.train.grad', param.grad, self.epoch_count)

        matrix_train = self.confusion_matrix_(self.list_to_tensor_list(self.predictions_train), self.list_to_tensor_list(self.actuals_train), False)
        true_neg, false_pos, false_neg, true_pos = matrix_train.ravel()
        self.plot_confusion_matrix(matrix_train, self.target_names, f"confusion_matrix_train_epoch_{self.epoch_count}")

        classification_report_train = classification_report(self.list_to_tensor_list(self.actuals_train), self.list_to_tensor_list(self.predictions_train), target_names=self.target_names, output_dict=True)
        trainDict = {
            'run' : self.run_count,
            'epoch' : self.epoch_count,
            'run_name' : self.utils.get_runtime(),
            'loss' : loss,
            'accuracy' : accuracy,
            'roc_auc' : roc_auc_train,
            'epoch_duration' : epoch_duration,
            'run_duration' : run_duration,
            'correct_prediction': self.epoch_num_correct,
            'predicted_alive_0': self.array_value_count(self.list_to_tensor_list(self.predictions_train, False), 0),
            'predicted_dead_1': self.array_value_count(self.list_to_tensor_list(self.predictions_train, False), 1),
            'true_pos': float(true_pos),
            'false_pos': float(false_pos),
            'false_neg': float(false_neg),
            'true_neg': float(true_neg),
            'precision_alive': classification_report_train['Alive']['precision'],
            'recall_alive': classification_report_train['Alive']['recall'],
            'f1_score_alive': classification_report_train['Alive']['f1-score'],
            'support_alive': classification_report_train['Alive']['support'],
            'precision_dead': classification_report_train['Dead']['precision'],
            'recall_dead': classification_report_train['Dead']['recall'],
            'f1_score_dead': classification_report_train['Dead']['f1-score'],
            'support_dead': classification_report_train['Dead']['support'],
            'accuracy_cr': classification_report_train['accuracy'],
            'early_stop': '-',
            'loss_reduced': '-',
            'model_saved': '-',
            'type': 'train',
            'activation_fn': self.utils.get_param('activation', '-'),
        }
        self.create_statistics(trainDict)
        if (self.utils.get_param('displayTable', True)):
            self.run_statistics()
        #self.tb.add_hparams(self.create_hparams(), self.create_stats_dict(trainDict))

        if (hasValidation):
            matrix_val = self.confusion_matrix_(self.list_to_tensor_list(self.predictions_validation), self.list_to_tensor_list(self.actuals_validation), False)
            true_neg, false_pos, false_neg, true_pos = matrix_val.ravel()
            self.plot_confusion_matrix(matrix_val, self.target_names, f"confusion_matrix_validation_epoch_{self.epoch_count}")

            classification_report_validation = classification_report(self.list_to_tensor_list(self.actuals_validation), self.list_to_tensor_list(self.predictions_validation), target_names=self.target_names, output_dict=True)
            validationDict = {
                'run' : self.run_count,
                'epoch' : self.epoch_count,
                'run_name' : self.utils.get_runtime(),
                'loss' : loss_validation,
                'accuracy' : accuracy_validation,
                'roc_auc' : roc_auc_validation,
                'epoch_duration' : epoch_validation_duration,
                'run_duration' : run_duration,
                'correct_prediction': self.epoch_validation_num_correct,
                'predicted_alive_0': self.array_value_count(self.list_to_tensor_list(self.predictions_validation, False), 0),
                'predicted_dead_1': self.array_value_count(self.list_to_tensor_list(self.predictions_validation, False), 1),
                'true_pos': float(true_pos),
                'false_pos': float(false_pos),
                'false_neg': float(false_neg),
                'true_neg': float(true_neg),
                'precision_alive': classification_report_validation['Alive']['precision'],
                'recall_alive': classification_report_validation['Alive']['recall'],
                'f1_score_alive': classification_report_validation['Alive']['f1-score'],
                'support_alive': classification_report_validation['Alive']['support'],
                'precision_dead': classification_report_validation['Dead']['precision'],
                'recall_dead': classification_report_validation['Dead']['recall'],
                'f1_score_dead': classification_report_validation['Dead']['f1-score'],
                'support_dead': classification_report_validation['Dead']['support'],
                'accuracy_cr': classification_report_validation['accuracy'],
                'early_stop': '-',
                'loss_reduced': '-',
                'model_saved': '-',
                'type': 'validation',
                'activation_fn': self.utils.get_param('activation', '-'),
            }
            if (self.estopping != None):
                dict = {
                'model' : self.network.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                'epoch': self.epoch_count,
                }
                self.estop = self.estopping.step(loss_validation, dict)
                validationDict['early_stop'] = 'Y' if self.estop else '-'
                validationDict['loss_reduced'] = self.estopping.best if self.estopping.loss_reduced else '-'
                validationDict['model_saved'] = 'Y' if self.estopping.model_saved else '-'
            self.create_statistics(validationDict)
            if (self.utils.get_param('displayTable', True)):
                self.run_statistics()
            #self.tb.add_hparams(self.create_hparams(), self.create_stats_dict(validationDict))
        if ((self.epoch_count % self.flush_tb_epoch) == 0):
            # self.tb.flush() # since tb flushes event in 120 second by default so this has been commented.
            self.utils.save_intermediate_summary_writer()

    def end_test(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        accuracy = self.epoch_num_correct / (len(self.loader.dataset)/3)
        roc_auc_test = roc_auc_score(self.list_to_tensor_list(self.actuals_test), self.list_to_tensor_list(self.predictions_test))
        
        self.tb.add_scalar('Accuracy-test', accuracy, self.epoch_count)
        self.tb.add_pr_curve('prec-rec-curve-test', self.list_to_tensor_list(self.actuals_test, False), self.list_to_tensor_list(self.predictions_original_test, False), self.epoch_count)

        matrix_test = self.confusion_matrix_(self.list_to_tensor_list(self.predictions_test), self.list_to_tensor_list(self.actuals_test), False)
        true_neg, false_pos, false_neg, true_pos = matrix_test.ravel()
        self.plot_confusion_matrix(matrix_test, self.target_names, f"confusion_matrix_test_epoch_{self.epoch_count}", predicted=True)

        classification_report_test = classification_report(self.list_to_tensor_list(self.actuals_test), self.list_to_tensor_list(self.predictions_test), target_names=self.target_names, output_dict=True)
        testDict = {
            'run' : self.run_count,
            'epoch' : self.epoch_count,
            'run_name' : self.utils.get_runtime(),
            'accuracy' : accuracy,
            'roc_auc' : roc_auc_test,
            'epoch_duration' : epoch_duration,
            'run_duration' : run_duration,
            'correct_prediction': self.epoch_num_correct,
            'predicted_alive_0': self.array_value_count(self.list_to_tensor_list(self.predictions_test, False), 0),
            'predicted_dead_1': self.array_value_count(self.list_to_tensor_list(self.predictions_test, False), 1),
            'true_pos': float(true_pos),
            'false_pos': float(false_pos),
            'false_neg': float(false_neg),
            'true_neg': float(true_neg),
            'precision_alive': classification_report_test['Alive']['precision'],
            'recall_alive': classification_report_test['Alive']['recall'],
            'f1_score_alive': classification_report_test['Alive']['f1-score'],
            'support_alive': classification_report_test['Alive']['support'],
            'precision_dead': classification_report_test['Dead']['precision'],
            'recall_dead': classification_report_test['Dead']['recall'],
            'f1_score_dead': classification_report_test['Dead']['f1-score'],
            'support_dead': classification_report_test['Dead']['support'],
            'accuracy_cr': classification_report_test['accuracy'],
            'type': 'test',
        }
        self.create_statistics(testDict)
        if (self.utils.get_param('displayTable', True)):
            self.run_statistics()

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
