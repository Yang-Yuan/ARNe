import glob

import numpy as np
import os
from tqdm import tqdm
import argparse


def main():

    # try:
    #     file = open(os.environ["CONFIGFILE"], "r")
    #     config = json.load(file)
    # except KeyError:
    #     try:
    #         file = open("../default_config.json", "r")
    #         config = json.load(file)
    #     except FileNotFoundError as e:
    #         raise e

    arg_parser = argparse.ArgumentParser(description = "piapiapia")
    arg_parser.add_argument("-s", type = str, required = True)
    arg_parser.add_argument("-d", type = str, required = True)
    args = arg_parser.parse_args()

    npz_path = args.s
    uint8_path = args.d

    for f in glob.glob(os.path.join(npz_path, "*.npz")):

        f = f.replace("\\", "/")
        npz = np.load(f)

        image = npz["image"]
        target = npz["target"]
        meta_target = npz["meta_target"]
        relation_structure_encoded = npz["relation_structure_encoded"]

        image_unit8 = image.astype(np.uint8)
        target_unit8 = target.astype(np.uint8)
        meta_target_unit8 = meta_target.astype(np.uint8)
        relation_structure_encoded_unit8 = relation_structure_encoded.astype(np.uint8)

        np.savez_compressed(os.path.join(uint8_path, f.split("/")[-1]),
                            image = image_unit8,
                            target = target_unit8,
                            meta_target = meta_target_unit8,
                            relation_structure_encoded = relation_structure_encoded_unit8)




if __name__ == "__main__":
    main()
