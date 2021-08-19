import datetime
import matplotlib
import torch.nn.functional as F
from datasets.PGM import *
from tensorboardX import SummaryWriter
import numpy as np

matplotlib.use('Agg')
import psutil

class Metrics:
	def __init__(self, model, len_dataset_train, len_dataset_val, lr, config, timestamp, bce=False, binary_label_size=None, test=False, len_dataset_test=None):
		# if no train mode is assigned it is set to None

		self.use_bce = bce

		self.use_relevance = True

		if bce:
			assert binary_label_size is not None
			self.binary_label_size = binary_label_size

		self.test_mode = False
		self.train_mode = False
		self.val_mode = False

		self.model = model
		self.config = config
		batch_size = self.config["dataset"]["dataloader"]["batch_size"]

		self.len_dataset_train = len_dataset_train
		self.len_dataset_val = len_dataset_val
		self.len_dataset_test = len_dataset_test

		self.N_batches_train = int(np.ceil(self.len_dataset_train / batch_size))
		self.N_batches_val = int(np.ceil(self.len_dataset_val / batch_size))

		if test:
			assert len_dataset_test is not None
			self.N_batches_test = int(np.ceil(self.len_dataset_test / batch_size))

		self.epoch = 0
		self.curr_batch = 0


		self.N_correct_predictions = 0
		self.N_correct_predictions_meta = 0
		self.total_dataset_size = 0

		self.accuracy_batch = 0
		self.accuracy_epoch = 0

		# average loss of targets
		self.loss = np.inf
		self.loss_total_epoch = 0

		# loss of targets
		self.loss_batch = np.inf
		self.data = {}

		if self.use_bce:
			self.tp = 0
			self.fp = 0
			self.tn = 0
			self.fn = 0

		self.modes = ["train", "val"]

		if test:
			self.modes.append("test")

		for mode in self.modes:
			self.data[mode] = {
								"losses": {"targets": {}},
								"accuracies": {"targets": {}},
								"relevance": {"targets": {}},
								}


		self.data["lrs"] = {
								1 : lr
		}

		self.lr = lr

		if self.config["visualisation"]["use_tensorboardX"]:
			self.writer = SummaryWriter(log_dir= os.path.join(
																self.config["experiment"]["experiments_path"],
																self.config["experiment"]["name"],
																"tensorboardX",
																"tensorboardX-{}".format(timestamp)))


		else:
			self.writer = None


	def get_loss(self, epoch, mode):
		assert epoch in self.data[mode]["losses"]["targets"].keys()
		return self.data[mode]["losses"]["targets"][epoch]

	def monitor_memory(self):
		self.data["memory_usage"].append(psutil.virtual_memory().used / 1024 ** 3)

	def update_writer_epoch(self):

		self.writer.add_scalar("learning_rate", self.lr, self.epoch)

		if self.config["model"]["train"] and self.config["model"]["val"]:
			for mode in ["train", "val"]:
				assert self.epoch in self.data[mode]["losses"]["targets"].keys()
				assert self.epoch in self.data[mode]["accuracies"]["targets"].keys()

			self.writer.add_scalars(
										"epoch/loss/target",

										{
											"Training": self.data["train"]["losses"]["targets"][self.epoch],
											"Validation": self.data["val"]["losses"]["targets"][self.epoch]
										},

										self.epoch
									)

			self.writer.add_scalars(
										"epoch/accuracy/target",

										{
											"Training": self.data["train"]["accuracies"]["targets"][self.epoch],
											"Validation": self.data["val"]["accuracies"]["targets"][self.epoch]
										},

										self.epoch
									)

			if self.use_bce:
				self.writer.add_scalars(
					"epoch/F1/target",

					{
						"Training": self.data["train"]["relevance"]["targets"][self.epoch]["F1"],
						"Validation": self.data["val"]["relevance"]["targets"][self.epoch]["F1"]
					},

					self.epoch
				)

	def update_writer_batch(self, global_step):
		"""
		Writes values to the tensorboardX Summary Writer
		:param global_step: N_batches*epoch + i_batch
		:return:
		"""
		if self.train_mode or self.val_mode:
			if self.train_mode:
				suffix = "train"

			else:
				suffix = "val"

			if self.writer is not None:
				self.writer.add_scalars("batch/loss/target/" + suffix,
										{
											"Batch": self.loss_batch,
											"Mean": self.loss
										},
											global_step
										)

				self.writer.add_scalars("batch/accuracy/target/" + suffix,
									{
										"Mean": self.accuracy_epoch,
										"Batch": self.accuracy_batch
									},
										global_step
									)


	def assert_data_dict(self, data):
		keys = self.modes
		keys.append("lrs")

		for key in data.keys():

			assert key in keys

			#if key == "train" or key == "val":
			if key != "lrs":
				assert False not in [metric in ["relevance", "accuracies", "losses"] for metric in data[key].keys()]
				assert "targets" in data[key]["accuracies"].keys()

	def load_data_dict(self, data):
		self.assert_data_dict(data)
		self.data = data

		# keys represent the epoch as integers. sorted returns a list of tuples where these tuples are sorted by its
		# first entry, the epoch. second entry is the learning rate. self.data["lrs"] stores only unique lrs

		self.lr = sorted(self.data["lrs"].items())[-1][1]


	def reset(self):
		self.epoch = 0
		self.curr_batch = 0
		self.total_dataset_size = 0

		# target
		self.N_correct_predictions = 0

		self.accuracy_batch = 0
		self.accuracy_epoch = 0

		self.loss = np.inf
		self.loss_total_epoch = 0

		self.loss_batch = np.inf

		if self.use_bce:
			self.tp = 0
			self.fp = 0
			self.tn = 0
			self.fn = 0

		self.train_mode = False
		self.test_mode = False
		self.val_mode = False

	def eval(self, epoch):
		self.reset()
		
		self.train_mode = False
		self.val_mode = True
		self.test_mode = False

		self.epoch = epoch


	def train(self, epoch):
		self.reset()

		self.train_mode = True
		self.val_mode = False
		self.test_mode = False
		self.epoch = epoch

	def test(self, epoch):
		self.reset()

		self.train_mode = False
		self.val_mode = False
		self.test_mode = True

		self.epoch = epoch

	def write_to_data(self, mode):
		self.data[mode]["losses"]["targets"][self.epoch] = self.loss
		self.data[mode]["accuracies"]["targets"][self.epoch] = self.accuracy_epoch

		if self.use_bce:
			self.data[mode]["relevance"]["targets"][self.epoch] = {"TP": self.tp,
																	"FP": self.fp,
																	"TN": self.tn,
																	"FN": self.fn,
																	"Precision": self.tp / (self.tp + self.fp),
																	"Recall": self.tp / (self.tp + self.fn),
			                                                       "F1": self.calculate_f_score(self.tp, self.fp, self.fn)
																	}

	def write(self):
		assert sum([self.train_mode, self.test_mode, self.val_mode]) == 1

		if self.train_mode:
			mode = "train"

		elif self.test_mode:
			mode = "test"

		elif self.val_mode:
			mode = "val"

		self.write_to_data(mode)

	def calculate_f_score(self, tp, fp, fn, beta=1):
		return (1 + beta**2) * tp / float((1 + beta**2)*tp + beta**2*fn + fp)


	def set_lr(self, lr, epoch):
		assert epoch not in self.data["lrs"].keys()
		self.data["lrs"][epoch] = lr
		self.lr = lr

	def _get_target_accuracy(self, logits_labels, correct_labels):

		if self.use_bce:
			# TODO optimise this. delete duplicates
			p_labels = torch.sigmoid(logits_labels)
			predictions = p_labels.ge(0.5).long()
			N_correct_batch = torch.eq(predictions.view_as(correct_labels), correct_labels.long()).float()

			# if self.use_relevance:
			self.get_relevance(N_correct_batch, correct_labels)

			N_correct_batch = N_correct_batch.sum()

			self.N_correct_predictions += N_correct_batch

			self.accuracy_epoch = float(100. * self.N_correct_predictions / (self.total_dataset_size * self.binary_label_size))
			self.accuracy_batch = float(100 * N_correct_batch / (self.current_batch_size * self.binary_label_size))


		else:
			p_labels = F.softmax(logits_labels, dim=1)
			predictions = torch.max(p_labels, dim=1)[1]
			N_correct_batch = torch.eq(correct_labels, predictions.view_as(correct_labels))

			N_correct_batch = N_correct_batch.sum().item()

			self.N_correct_predictions += N_correct_batch

			self.accuracy_epoch = float(100. * self.N_correct_predictions / self.total_dataset_size)
			self.accuracy_batch = float(100 * N_correct_batch / self.current_batch_size)


	def get_relevance(self, correct_predictions, correct_target):
		assert [target in torch.FloatTensor([0, 1]) for target in correct_predictions.unique()]
		correct_predictions = correct_predictions.byte()

		self.tn += torch.sum(torch.masked_select(correct_target, correct_predictions).eq(0)).item()
		self.fp += torch.sum(torch.masked_select(correct_target, ~ correct_predictions).eq(0)).item()

		self.tp += torch.sum(torch.masked_select(correct_target, correct_predictions).eq(1)).item()
		self.fn += torch.sum(torch.masked_select(correct_target, ~ correct_predictions).eq(1)).item()

	def _get_loss(self, loss_batch):
		self.loss_total_epoch += loss_batch * self.current_batch_size
		self.loss = self.loss_total_epoch / self.total_dataset_size


	def update(self, logits_labels, correct_labels, loss_batch):
		assert self.train_mode is not None
		# copy los and answer logits to cpu
		loss_batch = float(loss_batch.cpu().clone())
		logits_labels = logits_labels.cpu().clone()
		correct_labels = correct_labels.cpu().clone()

		self.curr_batch += 1

		self.current_batch_size = len(correct_labels)
		self.total_dataset_size += self.current_batch_size

		self._get_target_accuracy(logits_labels, correct_labels)
		self._get_loss(loss_batch)

		self.loss_batch = loss_batch



	def last_batch(self):
		assert sum([self.train_mode, self.test_mode, self.val_mode]) == 1

		if self.train_mode:
				N_batches = self.N_batches_train

		elif self.val_mode:
			N_batches = self.N_batches_val

		elif self.test_mode:
			N_batches = self.N_batches_test

		if self.curr_batch == N_batches:
			return True

		else:
			return False


	def print_status(self):
		assert sum([self.train_mode, self.test_mode, self.val_mode]) == 1

		if self.config["debug"]["log"]["print_metrics"] and (
				self.curr_batch % self.config["debug"]["log"]["print_after_n_batches"]) == 0:

			if self.train_mode:
				N_batches = self.N_batches_train

			elif self.test_mode:
				N_batches = self.N_batches_test

			elif self.val_mode:
				N_batches = self.N_batches_val

			self.print_batch_message(N_batches)

			# tensorboardX
			# CAUTION! THIS TAKES 3 SECONDS PER CALL AND WILL MASSIVELY SLOW DOWN YOUR TRAINING IF OVERUSED
			if self.writer is not None and self.curr_batch % self.config["debug"]["log"]["print_after_n_batches"]*10 == 0:
				global_step = N_batches * (self.epoch - 1) + self.curr_batch
				self.update_writer_batch(global_step)

		if self.last_batch():

			self.write()
			self.print_summary()

			if self.val_mode:
				self.update_writer_epoch()




	def get_notification_data(self, exp_name, timestamp):

		notification_message = "Checkpoint:\t{}\nBest Epoch:\t{}\n\nAccuracies\ntrain:\t{}\nval:\t\t{}\ntest:\t{}\n\nLosses\ntrain:\t{}\nval:\t\t{}\ntest:\t{}"

		notification_title = "{} finished"

		modes = {}

		for mode in ["train", "val", "test"]:
			modes[mode] = { "acc": self.data[mode]["accuracies"]["targets"][self.epoch],
							"loss":  self.data[mode]["losses"]["targets"][self.epoch]
						}


		data = {"title": notification_title.format(exp_name), "message": notification_message.format(
																			timestamp,
																			 self.epoch,
																			 modes["train"]["acc"],
																			 modes["val"]["acc"],
																			 modes["test"]["acc"],
																			 modes["train"]["loss"],
																			 modes["val"]["loss"],
																			 modes["test"]["loss"],
																			 )}

		return data

	def print_batch_message(self, N_batches):
		assert sum([self.train_mode, self.test_mode, self.val_mode]) == 1

		if self.train_mode:
			mode = "Training"
			mode_ = "train"

		elif self.val_mode:
			mode = "Validation"
			mode_ = "val"

		elif self.test_mode:
			mode ="Test"


		status = "{} Epoch {} | {} | Batch {:d}/{:d} | Batch Loss {:.4f} | <Loss> {:.4f} | <Accuracy> {:.4f} | lr {:.4e}".format(
				datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
				self.epoch,
				mode,
				self.curr_batch,
				N_batches,
				self.loss_batch,
				self.loss,
				self.accuracy_epoch,
				self.lr,
			)

		if self.use_bce:
			status += self.get_precision_recall_status()


		print(status)

	def get_precision_recall_status(self):
		try:
			precision = self.tp / (self.tp + self.fp)

		except ZeroDivisionError:
			precision = np.inf

		try:
			recall = self.tp / (self.tp + self.fn)

		except ZeroDivisionError:
			recall = np.inf

		try:
			F1 = self.calculate_f_score(self.tp, self.fp, self.fn)

		except ZeroDivisionError:
			F1 = 0

		return " | P {: 4f} | R {: 4f} | F1 {}".format(precision, recall, F1)

	def print_summary(self):
		assert sum([self.train_mode, self.test_mode, self.val_mode]) == 1

		if self.train_mode:
			pretty_mode = "Training"
			mode = "train"

		elif self.val_mode:
			pretty_mode = "Validation"
			mode = "val"

		elif self.test_mode:
			pretty_mode = "Testing"
			mode = "test"

		status = "Finished {}: Loss {:.4f} Accuracy {:.4f}".format(
														pretty_mode,
														self.data[mode]["losses"]["targets"][self.epoch],
														self.data[mode]["accuracies"]["targets"][self.epoch]
														)

		if self.use_bce:
			status += self.get_precision_recall_status()

		print(status)


class Metrics_WReN(Metrics):
	def __init__(self,  model, len_dataset_train, len_dataset_val, lr, config, timestamp, bce=False, binary_label_size=None, test=False, len_dataset_test=None):
		super().__init__(model, len_dataset_train, len_dataset_val, lr, config, timestamp, bce=bce, binary_label_size=binary_label_size, test=test, len_dataset_test=len_dataset_test)

		self.meta_labels_size = 12.
		self.N_correct_predictions_meta = 0

		self.accuarcy_meta_labels_batch = 0
		self.accuarcy_meta_labels_epoch = 0

		self.loss_meta = np.inf
		self.loss_meta_epoch = 0

		self.loss_meta_target = np.inf
		self.loss_meta_target_epoch = 0

		# relevance metrices
		self.tp_meta = 0
		self.fp_meta = 0
		self.tn_meta = 0
		self.fn_meta = 0

		modes = ["train", "val"]

		if test:
			modes.append("test")

		for mode in modes:
			self.data[mode]["accuracies"]["meta_targets"] = {}
			self.data[mode]["losses"]["loss_meta"] = {}
			self.data[mode]["losses"]["loss_meta_target"] = {}
			self.data[mode]["relevance"]["meta_targets"] = {}

		self.performance_template = {
										"relation": {
														"progression": -1,
														"XOR": -1,
														"OR": -1,
														"AND": -1,
														"consistent_union": -1
													},
										"object": {
													"shape": -1,
													"line": -1
													},
										"attribute": {
														"size": -1,
														"type": -1,
														"position": -1,
														"number": -1,
														"color": -1,
													}
									}

		
		# contains tp,fn,fp,fn per triple element
		self.roa_tp = torch.zeros(12)
		self.roa_tn = torch.zeros(12)
		self.roa_fp = torch.zeros(12)
		self.roa_fn = torch.zeros(12)

		self.roa_index = torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11], dtype=torch.uint8)
		


		self.n_meta_target_prediction = torch.zeros(12).long()
		self.n_meta_target = torch.zeros(12).long()

		# Encoding of meta_targets. Needed for perfomance measurements
		self.meta_targets_decoder = {0: "shape",
									 1: "line",
									 2: "color",
									 3: "number",
									 4: "position",
									 5: "size",
									 6: "type",
									 7: "progression",
									 8: "XOR",
									 9: "OR",
									 10: "AND",
									 11: "consistent_union"
									 }

	def get_loss(self, epoch, mode):
		assert epoch in self.data[mode]["losses"]["loss_meta_target"].keys()
		return self.data[mode]["losses"]["loss_meta_target"][epoch]

	def get_notification_data(self, exp_name, timestamp):
		notification_message = """Checkpoint:\t{}
					Best Epoch:\t{}
					\nAccuracies
					train:\t{:.4f}
					val:\t\t{:.4f}
					test:\t{:.4f}
					\nLosses: Target/Meta/Total
					train:\t{:.4f}\t{:.4f}\t{:.4f}
					val:\t\t{:.4f}\t{:.4f}\t{:.4f}
					test:\t{:.4f}\t{:.4f}\t{:.4f}
					\nMeta Accuracies
					train:\t{:.4f}
					val:\t\t{:.4f}
					test:\t{:.4f}
					\n F1 scores
					train:\t{:.4f}
					val:\t\t{:.4f}
					test:\t{:.4f}"""

		notification_title = "{} finished"

		modes = {}

		for mode in ["train", "val", "test"]:
			modes[mode] = { "acc": self.data[mode]["accuracies"]["targets"][self.epoch],
			                "acc_meta": self.data[mode]["accuracies"]["meta_targets"][self.epoch],
							"loss":  self.data[mode]["losses"]["targets"][self.epoch],
							"loss_meta": self.data[mode]["losses"]["loss_meta"][self.epoch],
							"loss_total": self.data[mode]["losses"]["loss_meta_target"][self.epoch],
							"f1_meta": self.data[mode]["relevance"]["meta_targets"][self.epoch]["F1"]
						}

		data = {"title": notification_title.format(exp_name), "message": notification_message.format(
																			timestamp,
																			self.epoch,
																			modes["train"]["acc"],
																			modes["val"]["acc"],
																			modes["test"]["acc"],
																			modes["train"]["loss"],
																			modes["train"]["loss_meta"],
																			modes["train"]["loss_total"],
																			modes["val"]["loss"],
																			modes["val"]["loss_meta"],
																			modes["val"]["loss_total"],
																			modes["test"]["loss"],
																			modes["test"]["loss_meta"],
																			modes["test"]["loss_total"],
																			modes["train"]["acc_meta"],
																			modes["val"]["acc_meta"],
																			modes["test"]["acc_meta"],
																			modes["train"]["f1_meta"],
																			modes["val"]["f1_meta"],
																			modes["test"]["f1_meta"],
																			)}

		return data


	def assert_data_dict(self, data):
		super().assert_data_dict(data)

		for key in data.keys():
			if key == "train" or key == "val" or key == "test":
				assert "meta_targets" in data[key]["accuracies"].keys()
				assert "loss_meta_target" in data[key]["losses"].keys() and "loss_meta" in data[key]["losses"].keys()
				assert "meta_targets" in data[key]["relevance"].keys()


	def write_to_data(self, mode):
		print("write_to_data_wren")
		super().write_to_data(mode)
		self.data[mode]["accuracies"]["meta_targets"][self.epoch] = self.accuarcy_meta_labels_epoch
		self.data[mode]["losses"]["loss_meta"][self.epoch] = self.loss_meta
		self.data[mode]["losses"]["loss_meta_target"][self.epoch] = self.loss_meta_target
		self.data[mode]["relevance"]["meta_targets"][self.epoch] = {"TP": self.tp_meta,
																	"FP": self.fp_meta,
																	"TN": self.tn_meta,
																	"FN": self.fn_meta,
																	"Precision": self.tp_meta / (self.tp_meta + self.fp_meta),
																	"Recall": self.tp_meta / (self.tp_meta + self.fn_meta),
																	"F1": super().calculate_f_score(self.tp_meta, self.fp_meta,
																								 self.fn_meta)
																	}

		# performance measures per relation/object/attribute
		assert self.n_meta_target_prediction.shape == (12,)
		assert self.n_meta_target.shape == (12,)

		n_meta_target_prediction = self.n_meta_target_prediction.float().numpy()
		n_meta_target = self.n_meta_target.float().numpy()


		rel = ["progression", "XOR", "OR", "AND", "consistent_union"]
		obj = ["shape", "line"]
		att = ["size", "type", "position", "number", "color"]


		performance = {
			"relation": {
				"progression": -1,
				"XOR": -1,
				"OR": -1,
				"AND": -1,
				"consistent_union": -1
			},
			"object": {
				"shape": -1,
				"line": -1
			},
			"attribute": {
				"size": -1,
				"type": -1,
				"position": -1,
				"number": -1,
				"color": -1,
			}
		}

		for i in range(12):
			key = self.meta_targets_decoder[i]

			if key in rel:
				s_type = "relation"

			if key in obj:
				s_type = "object"

			if key in att:
				s_type = "attribute"

			performance[s_type][key] = {"acc": n_meta_target_prediction[i]/n_meta_target[i]*100, "n_prediction": n_meta_target_prediction[i], "n_target": n_meta_target[i]}

			if self.test_mode:
				performance[s_type][key]["TP"] = self.roa_tp[i].item()
				performance[s_type][key]["TN"] = self.roa_tn[i].item()
				performance[s_type][key]["FP"] = self.roa_fp[i].item()
				performance[s_type][key]["FN"] = self.roa_fn[i].item()

				
				# tn rate
				try:
					performance[s_type][key]["TNR"] = performance[s_type][key]["TN"] / (performance[s_type][key]["TN"] + performance[s_type][key]["FP"])

				except ZeroDivisionError:
					performance[s_type][key]["TNR"] = -1

				# * fn rate
				try:
					performance[s_type][key]["FNR"] = performance[s_type][key]["FN"] / (performance[s_type][key]["FN"] + performance[s_type][key]["TP"])

				except ZeroDivisionError:
					performance[s_type][key]["FNR"] = -1
				# tp rate
				try:
					performance[s_type][key]["Recall"] = performance[s_type][key]["TP"] / (performance[s_type][key]["TP"] + performance[s_type][key]["FN"])

				except ZeroDivisionError:
					performance[s_type][key]["Recall"] = -1

				# * fp rate
				try:
					performance[s_type][key]["FPR"] = performance[s_type][key]["FP"] / (performance[s_type][key]["FP"] + performance[s_type][key]["TN"])

				except ZeroDivisionError:
					performance[s_type][key]["FPR"] = -1

				try:
					performance[s_type][key]["Precision"] =  performance[s_type][key]["TP"] / (performance[s_type][key]["TP"] + performance[s_type][key]["FP"])

				except ZeroDivisionError:
					performance[s_type][key]["Precision"] = -1

				try:
					performance[s_type][key]["F1"] = super().calculate_f_score(performance[s_type][key]["TP"],
																			   performance[s_type][key]["FP"],
																			   performance[s_type][key]["FN"])
				except ZeroDivisionError:
					performance[s_type][key]["F1"] = -1


		self.data[mode]["relevance"]["meta_targets"][self.epoch]["performance"] = performance

	def get_relevance_meta(self, correct_predictions, correct_target):
		"""
		example
		correct_predictions is a list holding 12 bits. if the target was predicted correctly, the i-th entry is equal to 1.

		target
		tensor([1, 1, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.uint8)
		predictions
		tensor([1, 1, 1, 1, 0, 0, 1, 1, 0, 0], dtype=torch.uint8)
		correct_predictions
		tensor([1, 1, 0, 0, 0, 0, 1, 1, 0, 0], dtype=torch.uint8)

		not correct_predictions
		tensor([0, 0, 1, 1, 1, 1, 0, 0, 1, 1], dtype=torch.uint8)
		TN
		tensor([0, 0, 0, 0], dtype=torch.uint8)
		0
		FP
		tensor([1, 1, 0, 0, 0, 0], dtype=torch.uint8)
		2
		TP
		tensor([1, 1, 1, 1], dtype=torch.uint8)
		4
		FN
		tensor([0, 0, 1, 1, 1, 1], dtype=torch.uint8)
		4
		precision
		0.6666666666666666
		recall
		0.5


		False positive: correct_target 0 -> prediction 1
		False negative: correct_target 1 -> prediction 0

		"""

		# binary metrics
		correct_predictions = correct_predictions.bool()

		masked_correct_predictions = torch.masked_select(correct_target, correct_predictions)
		masked_not_correct_predictions = torch.masked_select(correct_target, ~ correct_predictions)


		# 12 bit structure
		tns = masked_correct_predictions.eq(0)
		self.tn_meta += torch.sum(tns).item()
		
		fps = masked_not_correct_predictions.eq(0)
		self.fp_meta += torch.sum(fps).item()

		tps = masked_correct_predictions.eq(1)
		self.tp_meta += torch.sum(tps).item()
		
		fns = masked_not_correct_predictions.eq(1)
		self.fn_meta += torch.sum(fns).item()

		#roa
		if self.test_mode:
			self.roa_relevance(correct_predictions, tps, tns, fps, fns)

		# get relation/object/attribute accuracies
		self.n_meta_target_prediction += (correct_predictions.byte() * correct_target.byte()).sum(dim=0).long()
		self.n_meta_target += correct_target.sum(dim=0).long()

	def roa_relevance(self, correct_predictions, tps, tns, fps, fns):
		roa_index_correct = torch.masked_select(self.roa_index, correct_predictions)
		roa_index_not_correct = torch.masked_select(self.roa_index, ~correct_predictions)

		# tp
		roa_tps= torch.masked_select(roa_index_correct, tps)

		# tn
		roa_tns= torch.masked_select(roa_index_correct, tns)

		# fp
		roa_fps= torch.masked_select(roa_index_not_correct, fps)

		# fn
		roa_fns= torch.masked_select(roa_index_not_correct, fns)

		for _ in roa_tps.long():
			self.roa_tp[_] += 1

		for _ in roa_tns.long():
			self.roa_tn[_] += 1

		for _ in roa_fps.long():
			self.roa_fp[_] += 1

		

		for _ in roa_fns.long():
			self.roa_fn[_] += 1


	def _get_meta_target_accuracy(self, logits_meta_labels, correct_meta_labels):
		p_meta_labels = torch.sigmoid(logits_meta_labels)
		predictions = p_meta_labels.ge(0.5).long()


		correct_labels = torch.eq(predictions.view_as(correct_meta_labels), correct_meta_labels.long()).float()


		if self.use_relevance:
			self.get_relevance_meta(correct_labels, correct_meta_labels)

		correct_labels = correct_labels.sum()
		self.N_correct_predictions_meta += correct_labels

		self.accuarcy_meta_labels_batch = float(
			100 * correct_labels / (self.current_batch_size * self.meta_labels_size))
		self.accuarcy_meta_labels_epoch = float(
			100 * self.N_correct_predictions_meta / (self.total_dataset_size * self.meta_labels_size))


	def reset(self):
		super().reset()
		# Meta
		self.N_correct_predictions_meta = 0

		self.accuarcy_meta_labels_batch = 0
		self.accuarcy_meta_labels_epoch = 0

		self.loss_meta = np.inf
		self.loss_meta_epoch = 0

		self.loss_meta_target = np.inf
		self.loss_meta_target_epoch = 0

		# relevance metrices
		self.tp_meta = 0
		self.fp_meta = 0
		self.tn_meta = 0
		self.fn_meta = 0

		self.roa_tp = torch.zeros(12)
		self.roa_tn = torch.zeros(12)
		self.roa_fp = torch.zeros(12)
		self.roa_fn = torch.zeros(12)

		# counters for meta targets
		self.n_meta_target_prediction = torch.zeros(12).long()
		self.n_meta_target = torch.zeros(12).long()

	def update(self, logits_labels, logits_meta_labels, correct_labels, correct_meta_labels, loss_target_batch, loss_meta_batch, loss_meta_target_batch):
		super().update(logits_labels, correct_labels, loss_target_batch)

		logits_meta_labels = logits_meta_labels.cpu().clone()
		correct_meta_labels = correct_meta_labels.cpu().clone()

		loss_meta_batch = float(loss_meta_batch.cpu().clone())
		loss_meta_target_batch = float(loss_meta_target_batch.cpu().clone())

		self._get_meta_target_accuracy(logits_meta_labels, correct_meta_labels)
		self.get_meta_loss(loss_meta_batch, loss_meta_target_batch)

		self.loss_meta_target_batch = loss_meta_target_batch
		self.loss_meta_batch = loss_meta_batch

	def get_meta_loss(self, loss_meta_batch, loss_meta_target_batch):
		# there is * self.meta_labels_size missing. because it cancels out self.loss_meta this factor is ommitted
		self.loss_meta_epoch += loss_meta_batch * self.current_batch_size
		self.loss_meta = self.loss_meta_epoch / self.total_dataset_size

		self.loss_meta_target_epoch += loss_meta_target_batch * self.current_batch_size
		self.loss_meta_target = self.loss_meta_target_epoch / self.total_dataset_size

	def update_writer_batch(self, global_step):
		super().update_writer_batch(global_step)

		if self.train_mode or self.val_mode:
			if self.train_mode:
				suffix = "train"

			elif self.val_mode:
				suffix = "val"

			if self.writer is not None:
				self.writer.add_scalars("accuracy/batch/meta/" + suffix,
									{
										"Mean": self.accuarcy_meta_labels_epoch,
										"Batch": self.accuarcy_meta_labels_batch
									},
										global_step
									)
				self.writer.add_scalars("loss/batch/meta/" + suffix,
											{
												"Mean": self.loss_meta,
												"Batch": self.loss_meta_batch
											},
												global_step
										)

				self.writer.add_scalars("loss/batch/target_and_meta/" + suffix,
											{
												"Mean": self.loss_meta_target,
												"Batch": self.loss_meta_target_batch
											},
												global_step
										)

	def update_writer_epoch(self):
		super().update_writer_epoch()

		self.writer.add_scalars(
			"epoch/loss/target_and_meta",

			{
				"Training": self.data["train"]["losses"]["loss_meta_target"][self.epoch],
				"Validation": self.data["val"]["losses"]["loss_meta_target"][self.epoch]
			},

			self.epoch
		)

		self.writer.add_scalars(
			"epoch/loss/meta",

			{
				"Training": self.data["train"]["losses"]["loss_meta"][self.epoch],
				"Validation": self.data["val"]["losses"]["loss_meta"][self.epoch]
			},

			self.epoch
		)

		self.writer.add_scalars(
			"epoch/accuracy/target_and_meta",

			{
				"Training": self.data["train"]["accuracies"]["meta_targets"][self.epoch],
				"Validation": self.data["val"]["accuracies"]["meta_targets"][self.epoch]
			},

			self.epoch
		)

		self.writer.add_scalars(
			"epoch/F1/meta_target",

			{
				"Training": self.data["train"]["relevance"]["meta_targets"][self.epoch]["F1"],
				"Validation": self.data["val"]["relevance"]["meta_targets"][self.epoch]["F1"]
			},

			self.epoch
		)


	def get_precision_recall_status_meta(self):
		try:
			precision = self.tp_meta / (self.tp_meta + self.fp_meta)

		except ZeroDivisionError:
			precision = np.inf

		try:
			recall = self.tp_meta / (self.tp_meta + self.fn_meta)

		except ZeroDivisionError:
			recall = np.inf

		try:
			F1 = super().calculate_f_score(self.tp_meta, self.fp_meta, self.fn_meta)

		except ZeroDivisionError:
			F1 = 0

		return " | P {: 4f} | R {: 4f} | F1 {}".format(precision, recall, F1)

	def print_batch_message(self, N_batches):
		assert sum([self.train_mode, self.test_mode, self.val_mode]) == 1

		if self.train_mode:
			mode = "Training"

		elif self.val_mode:
			mode = "Validation"

		elif self.test_mode:
			mode = "Testing"

		status = "{} Epoch {} | {} | Batch {:d}/{:d} | Batch Loss {:.4f} | <Loss> {:.4f} | <Accuracy> {:.4f} | <Accuracy Meta> {:.4f} | lr {:.4e}".format(
				datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
				self.epoch,
				mode,
				self.curr_batch,
				N_batches,
				self.loss_meta_target_batch,
				self.loss_meta_target,
				self.accuracy_epoch,
				self.accuarcy_meta_labels_epoch,
				self.lr,
			)

		if self.use_relevance:
			status += self.get_precision_recall_status_meta()

		print(status)

	def print_summary(self):
		assert sum([self.train_mode, self.test_mode, self.val_mode]) == 1

		if self.train_mode:
			pretty_mode = "Training"
			mode = "train"

		elif self.val_mode:
			pretty_mode = "Validation"
			mode = "val"

		elif self.test_mode:
			pretty_mode = "Testing"
			mode = "test"

		status = "Finished {}: Loss {:.4f} Accuracy {:.4f} Accuracy Meta {:.4f}".format(
			pretty_mode,
			self.data[mode]["losses"]["targets"][self.epoch],
			self.data[mode]["accuracies"]["targets"][self.epoch],
			self.data[mode]["accuracies"]["meta_targets"][self.epoch]
			)

		if self.use_relevance:
			status += self.get_precision_recall_status_meta()

		print(status)
