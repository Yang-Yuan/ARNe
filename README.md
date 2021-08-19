ARNe
----
Source Code of the Attention on Abstract Visual Reasoning [Paper](https://arxiv.org/pdf/1911.05990.pdf)

Setup Python
------------
- This project uses `PyTorch v 1.1.0`. Check your Python Interpreter before you proceed.
- In the root path of this project run

``` bash
python3 -m venv arne-env
source arne-env/bin/activate
pip install -r requirements.txt
```

If you want to run on one or more GPU/GPUs change the `CUDA_VISIBLE_DEVICES` accordingly. Default is set to CPU.

```bash
./run.sh
```

will run the model with options used in ARNe.

Data
-----
- Get your copy [here](https://console.cloud.google.com/storage/browser/ravens-matrices)
- The entire dataset is encoded in int64, 8 byte Integers. However, uint8 bytes are sufficient and accelerates learning. 
Run `setup_dataset.sh` to cast to uints and store the resulting datastructures under the `pt_path` path specified in the config file

Config File
-----------
- Before you get started you have to set following paths in `default_config.json`:
  - `npz_path`: path to the downloaded dataset
  - `pt_path`: target directory of processed dataset which will be used in this model
- Make sure to set `num_workers=0` when computing on CPU



Visualisation
-------------
You can monitor and visualise key metrics via tensorboard:

```bash
tensorboard --logdir=./experiments/<experiment_name>/tensorboardX/tensorboardX-<date>
```
