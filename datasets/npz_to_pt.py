import numpy as np
import os
from tqdm import tqdm
import torch
import json

def main():

	with open(os.environ["CONFIGFILE"], "r") as file:
		config = json.load(file)


	npz_path = config["dataset"]["npz_path"]
	pt_path = config["dataset"]["pt_path"]

	filename_template_npz = "PGM_{}_{}_{}"
	filename_template = "{}_" + filename_template_npz

	pt_file = os.path.join(pt_path, filename_template + ".pt")
	npz_file = os.path.join(npz_path, filename_template_npz + ".npz")

	dataset_type = "neutral"

	if not os.path.isdir(pt_path):
		os.mkdir(pt_path)


	size = {"train": 1200000, "val":20000, "test": 200000}


	for mode in ["train", "val", "test"]:

		for i in tqdm(range(1, size[mode] + 1), desc=mode):
			npz = np.load(os.path.join(npz_path, npz_file.format(dataset_type, mode, i)))

			image = npz["image"]
			target = npz["target"]
			meta_target = npz["meta_target"]

			assert (not np.min(image) < 0 and not np.max(image) > 255) and (not np.min(target) < 0 and not np.max(target) > 255) and (not np.min(meta_target) < 0 and not np.max(meta_target) > 255)


			relation_structure_encoded = npz["relation_structure_encoded"]
			image_torch = torch.from_numpy(image.astype(np.uint8)).reshape(16, 160, 160)

			torch.save(image_torch, pt_file.format("image", dataset_type, mode, i))
			torch.save(torch.from_numpy(target.astype(np.uint8)), pt_file.format("target", dataset_type, mode, i))
			torch.save(torch.from_numpy(meta_target.astype(np.uint8)), pt_file.format("meta_target", dataset_type, mode, i))
			torch.save(torch.from_numpy(relation_structure_encoded.astype(np.uint8)),
			 		pt_file.format("relation_structure_encoded", dataset_type,  mode, i))


if __name__ == "__main__":
	main()