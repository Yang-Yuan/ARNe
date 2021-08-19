import os
import torch
from torch.utils import data

from enum import Enum

class PGMType(Enum):
	train = "train"
	val = "val"
	test = "test"



class PGM(data.Dataset):
	def __init__(self, pgmtype, data_path):

		if pgmtype not in list(PGMType):
			raise ValueError("No such PGM type: %s" % pgmtype)

		self.data_path = data_path

		base_file_name = "_PGM_neutral_" + pgmtype.value + "_{}.pt"

		if not isinstance(pgmtype, PGMType):
			raise ValueError("Please use PGMType")

		self.pgmtype = pgmtype


		self.image_file = os.path.join(self.data_path, "image" + base_file_name)
		self.target_file = os.path.join(self.data_path, "target" + base_file_name)
		self.meta_target_file = os.path.join(self.data_path, "meta_target" + base_file_name)


	def __len__(self):
		if self.pgmtype == PGMType.train:
			return 1200000


		elif self.pgmtype == PGMType.val:
			return 20000


		elif self.pgmtype == PGMType.test:
			return 200000


	def __getitem__(self, idx):

		idx += 1

		images = torch.load(self.image_file.format(idx))
		meta_target = torch.load(self.meta_target_file.format(idx))
		target = torch.load(self.target_file.format(idx))

		return images, target, meta_target, idx

	@staticmethod
	def cast_data(images, target, meta_target):
		target = target.long()
		meta_target = meta_target.float()

		batch_size = images.shape[0]
		images = images.float()

		images_context = images[:, :8, :, :, ]
		images_choices = images[:, 8:, :, :, ]

		return images_context, images_choices, target, meta_target