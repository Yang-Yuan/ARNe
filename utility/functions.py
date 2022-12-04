import datetime
import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data


class Learning_Rate_Scheduler:
	def __init__(self, d_model, warm_up_steps, optimiser):
		self.warm_up_steps = warm_up_steps
		self.current_step = 0
		self.lr_init = np.power(d_model, -0.5)
		self.optimiser = optimiser

	def update_lr(self):
		self.current_step += 1
		lr = self.lr_init * np.min(
			[np.power(self.current_step, -0.5), self.current_step * np.power(self.warm_up_steps, -1.5)])

		for param_group in self.optimiser.param_groups:
			param_group['lr'] = lr

	def get_parameters(self):
		return {"warm_up_steps": self.warm_up_steps, "current_step": self.current_step, "lr_init": self.lr_init}

	def restore_parameters(self, param_dict):
		assert False not in [x in param_dict.keys() for x in ["warm_up_steps", "current_step", "lr_init"]]

		self.warm_up_steps = param_dict["warm_up_steps"]
		self.current_step = param_dict["current_step"]
		self.lr_init = param_dict["lr_init"]


class Early_Stopping:
	"""
    This algorithm is based on
    https://www.deeplearningbook.org/contents/regularization.html p 244
    """

	def __init__(self, patience, timestamp, dirs):

		self.epoch_best = 0
		self.j = 0

		# diff from algorithm. measure accuracy not error
		self.acc_best = 0

		self.best_epoch = 0

		self.p = patience
		self.stop = False

		self.timestamp = timestamp
		self.dirs = dirs

	def check(self, model, optimizer, ema, metrics, epoch, lr_scheduler = None):
		val_acc = metrics.data["val"]["accuracies"]["targets"][epoch]

		if self.j < self.p:
			if self.acc_best < val_acc:
				self.acc_best = val_acc
				self.epoch_best = epoch

				save_checkpoint(epoch, model, optimizer, metrics, self.dirs, self.timestamp, ema,
								lr_scheduler = lr_scheduler)
				self.j = 0
			else:
				self.j += 1
		else:
			self.stop = True

		print(
			"Early Stopping> best acc {} curr acc {} | j {} p {} | best epoch {}".format(self.acc_best, val_acc, self.j,
																						 self.p, self.epoch_best))

	def save_no_condition(self, model, optimizer, ema, metrics, epoch):
		print("Saving Checkpoint")
		save_checkpoint(epoch, model, optimizer, metrics, self.dirs, self.timestamp, ema)


def restore(model, metrics, ema, dirs, config, rename_model_layer = None, use_lr_scheduler = False):
	print("Restoring ...")
	checkpoint = load_checkpoint(dirs["checkpoints"], config["experiment"]["checkpoint"]["timestamp"])

	# if model was transfered to multiple gpus
	if torch.cuda.device_count() > 1:
		from collections import OrderedDict
		states_parallel = OrderedDict()

		for k, v in checkpoint["model_state_dict"].items():
			if 'module' not in k:
				k = 'module.' + k

			states_parallel[k] = v
			checkpoint["model_state_dict"] = states_parallel

	if rename_model_layer is not None:
		checkpoint = rename_model_layer(checkpoint, rename_model_layer["old_layer"], rename_model_layer["new_layer"])

	# iteration starts at NEXT epoch
	epoch_start = checkpoint['epoch'] + 1
	model.load_state_dict(checkpoint["model_state_dict"])

	for name, parameter in model.named_parameters():
		assert checkpoint["model_state_dict"][name].equal(parameter)

	optimiser = torch.optim.Adam(model.parameters(), lr = config["optimisation"]["optimiser"]["learning_rate"])
	optimiser.load_state_dict(checkpoint["optimizer_state_dict"])

	for key in optimiser.state_dict()["param_groups"][0].keys():
		print("key {} {}".format(key, checkpoint["optimizer_state_dict"]["param_groups"][0][key] ==
								 optimiser.state_dict()["param_groups"][0][key]))

	metrics.load_data_dict(checkpoint["metrics"])

	if config["optimisation"]["ema"]["use_ema"]:
		ema.restore_ema_parameters(checkpoint["ema"])

	if use_lr_scheduler:
		assert "lr_scheduler" in checkpoint.keys()
		# 0 -> dummy init val
		lr_scheduler = Learning_Rate_Scheduler(1, 1, optimiser)
		lr_scheduler.restore_parameters(checkpoint["lr_scheduler"])

		return model, optimiser, ema, epoch_start, metrics, lr_scheduler

	else:
		return model, optimiser, ema, epoch_start, metrics


def move_to_devices(model):
	cuda_devices = torch.cuda.device_count()

	if cuda_devices > 0:
		print("Using cuda.")
		model = model.cuda()

		if cuda_devices > 1:
			print("Using DataParallel")
			model = nn.DataParallel(model)

	return model


def create_timestamp():
	"""
    :return: String formatted time stamp of current time
    """
	return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def create_dirs(timestamp, config):
	"""
    Creates directories of the experiments, checkpoints, logs, configs, and tensorboardX
    :param timestamp: String formatted timestamp
    :return: Dictionary of paths
    """
	paths = {}

	path_experiment = os.path.join(config["experiment"]["experiments_path"], config["experiment"]["name"])

	paths["experiment"] = path_experiment

	# init
	if not os.path.exists(path_experiment):
		os.makedirs(paths["experiment"])

	# dynamically add sudirectories directories
	for key in ["checkpoints", "configs", "tensorboardX"]:
		path = os.path.join(paths["experiment"], key)
		paths[key] = path

		if (not (key == "tensorboardX")) or ((key == "tensorboardX") and config["visualisation"]["use_tensorboardX"]):
			if not os.path.exists(path):
				os.makedirs(paths[key])

	return paths


def load_config_file(config_file, string = False):
	"""
    Loads the config file from disk given the parsed user input of the path of the configuration file
    :param config_file: Path to the configuration file
    :return: Loaded Config file as a dictionary.
    """
	assert config_file is not None

	if type(config_file) == list:
		config_file = config_file[0]

	global config

	if string:
		config = json.loads(config_file)

	else:
		print("loading config file {}".format(config_file))
		with open(config_file, "r") as file:
			config = json.load(file)

	return config


def load_checkpoint(checkpoint_path, checkpoint_timestamp, checkpoint_name = "checkpoint-{}"):
	"""
    Loads a checkpoint from disk
    :param checkpoint_path: Path of the checkpoint
    :param checkpoint_timestamp: The timestamp of the checkpoint
    :param checkpoint_name: Name of the checkpoint
    :return: Loaded checkpoint
    """

	# load from other experiment --> abs path to checkpoint
	if os.path.isfile(checkpoint_timestamp):
		checkpoint_path = checkpoint_timestamp

	# otherwise load from same experiment given only the timestamp of the checkpoint
	else:
		checkpoint_path = os.path.join(checkpoint_path, checkpoint_name.format(checkpoint_timestamp))

	print("Loading checkpoint")
	n_cuda_devices = torch.cuda.device_count()

	if n_cuda_devices == 0:
		checkpoint = torch.load(checkpoint_path, map_location = "cpu")

	elif n_cuda_devices == 1:
		checkpoint = torch.load(checkpoint_path, map_location = "cuda:0")

	else:
		checkpoint = torch.load(checkpoint_path)

	print("Loaded checkpoint {}".format(checkpoint_path))

	return checkpoint


def save_checkpoint(epoch, model, optimizer, metrics, dirs, timestamp, ema, lr_scheduler = None):
	"""
    Saves data, including current epoch, state dict of model and optimiser and metrics,
    as a dictionary to disk.
    :param epoch: Current epoch
    :param model: The current model
    :param optimizer: The current optimiser
    :param metrics: Global metrics dictionary
    :param dirs: Dirs dictionary including the file path where to save the checkpoint
    :param timestamp: String formatted timestamp of the current run
    :param ema: EMA object or None
    :return:
    """
	print("Saving Checkpoint")

	if type(model) == nn.DataParallel:
		model_state_dict = model.module.state_dict()

	else:
		model_state_dict = model.state_dict()

	# model_state_dict are the best model parameters so far at given epoch
	checkpoint = {
		'epoch': epoch,
		'model_state_dict': model_state_dict,
		'optimizer_state_dict': optimizer.state_dict(),
		# 'metrics': metrics,
		'metrics': metrics.data,
	}

	if ema is not None:
		checkpoint["ema"] = ema.get_ema_parameters()

	if lr_scheduler is not None:
		checkpoint["lr_scheduler"] = lr_scheduler.get_parameters()

	torch.save(checkpoint, os.path.join(dirs["checkpoints"], "checkpoint-{}".format(timestamp)))

	# check if state was written correctly
	checkpoint = torch.load(os.path.join(dirs["checkpoints"], "checkpoint-{}".format(timestamp)))
	state_dict = checkpoint["model_state_dict"]
	state_dict_opt = checkpoint["optimizer_state_dict"]

	if type(model) == nn.DataParallel:
		for name, parameter in model.module.named_parameters():
			assert state_dict[name].equal(parameter)
	else:
		for name, parameter in model.named_parameters():
			assert state_dict[name].equal(parameter)

	for key in optimizer.state_dict()["param_groups"][0].keys():
		assert state_dict_opt["param_groups"][0][key] == optimizer.state_dict()["param_groups"][0][
			key]

	print("SAVE CHECKPOINT > PARAM CHECK OK")


def get_dataloader(data_set, collate_fn = None, shuffle = None):
	"""
	Creates a dataloader for a given data set.
	:param data_set: Name of the dataset
	:param collate_fn: Function to use on batch
	:return: a configured generator with respect to a given data set
	"""
	print("Initialising DataLoader")

	if shuffle is None:
		shuffle = config["dataset"]["dataloader"]["shuffle"]

	if collate_fn is not None:
		dataloader = data.DataLoader(data_set, batch_size = config["dataset"]["dataloader"]["batch_size"],
									 shuffle = shuffle,
									 num_workers = config["dataset"]["dataloader"]["num_workers"],
									 pin_memory = config["dataset"]["dataloader"]["pin_memory"],
									 collate_fn = collate_fn)
	else:
		dataloader = data.DataLoader(data_set, batch_size = config["dataset"]["dataloader"]["batch_size"],
									 shuffle = shuffle,
									 num_workers = config["dataset"]["dataloader"]["num_workers"],
									 pin_memory = config["dataset"]["dataloader"]["pin_memory"])

	return dataloader


def learning_rate_not_improved(curr_train_loss, prev_train_loss, curr_lr, thr_1 = 0.5, thr_2 = 0.15, thr_3 = 0.10):
	"""
    Checks if learning rate has not changed after each epoch of training

    :param curr_train_loss: The average loss of the whole epoch during training
    :param prev_train_loss: The average loss of the previous training session
    :param curr_lr: Current learning rate of the ADAM optimiser
    :return: boolean if learning rate has not improved
    """

	if prev_train_loss == None:
		return False

	loss_diff = np.abs(curr_train_loss - prev_train_loss)

	not_improved = ((loss_diff < 0.015 and prev_train_loss < thr_1 and curr_lr > 0.00002) or \
					(loss_diff < 0.008 and prev_train_loss < thr_2 and curr_lr > 0.00001) or \
					(loss_diff < 0.003 and prev_train_loss < thr_3 and curr_lr > 0.000005))

	return not_improved


def check_LR(config, metrics, optimiser, i_epoch, thr_1 = 0.5, thr_2 = 0.15, thr_3 = 0.10):
	"""
    Checks if learning rate has not changed and updates it

    :param config: The Configuration File
    :param metrics: The global metrics dictionary
    :param optimiser: Optimiser object holding the current learning rate
    :param i_epoch: Current epoch
    :return: updated global metrics dictionary
    """
	# reduce learning rate if loss does not improve
	if i_epoch > 1:
		prev_train_loss = metrics.get_loss(i_epoch - 1, "train")


	else:
		# during init
		prev_train_loss = None

	if config["optimisation"]["learning_rate_decay"]["use"] and learning_rate_not_improved(
			metrics.get_loss(i_epoch, "train"), prev_train_loss, optimiser.param_groups[0]["lr"], thr_1 = thr_1,
			thr_2 = thr_2, thr_3 = thr_3):
		prev_lr = optimiser.param_groups[0]["lr"]
		lr_decay = optimiser.param_groups[0]["lr"] * config["optimisation"]["learning_rate_decay"]["rate"]

		optimiser.param_groups[0]["lr"] = lr_decay

		assert optimiser.param_groups[0]["lr"] == prev_lr * config["optimisation"]["learning_rate_decay"]["rate"]

		print("Changed lr from {} to {}".format(prev_lr, optimiser.param_groups[0]["lr"]))
		metrics.set_lr(lr_decay, i_epoch + 1)


def copy_config_file(config_file, dirs, timestamp):
	with open(os.path.join(dirs["configs"], "config-{}".format(timestamp)), "w") as file:
		json.dump(config_file, file)
