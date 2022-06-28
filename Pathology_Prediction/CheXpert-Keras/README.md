# CheXpert-Keras
This project is a tool to build CheXpert-like models, written in Keras.

## What is [CheXpert](https://ojs.aaai.org/index.php/AAAI/article/download/3834/3712)?
CheXpert is a large dataset of chest X-rays and competition for automated chest x-ray interpretation, which features uncertainty labels and radiologist-labeled reference standard evaluation sets.

## In this project, you can
1. Train/test a **baseline model** by following the quickstart. You can get a model with performance close to the paper.
2. Run class activation mapping to see the localization of your model.
3. Modify `multiply` parameter in `config.ini` or design your own class weighting to see if you can get better performance.
4. Modify `weights.py` to customize your weights in loss function. If you find something useful, feel free to make that an option and fire a PR.
5. Every time you do a new experiment, make sure you modify `output_dir` in `config.ini` otherwise previous training results might be overwritten. For more options check the parameter description in `config.ini`.

## Quickstart
**Note that currently this project can only be executed in Linux and macOS. You might run into some issues in Windows.**
1. Download **all tar files**, **train.csv** and **valid.csv** of CheXpert dataset from [Stanford Mirror](https://stanfordmlgroup.github.io/competitions/chexpert/). Put them under `./data/default_split` folder and untar all tar files.
2. Create & source a new virtualenv. Python >= **3.6** is required.
3. Install dependencies by running `pip3 install -r requirements.txt`.
4. Copy sample_config.ini to config.ini, you may customize `batch_size` and training parameters here. Make sure config.ini is configured before you run training or testing
5. Run `python train.py` to train a new model. If you want to run the training using multiple GPUs, just prepend `CUDA_VISIBLE_DEVICES=0,1,...` to restrict the GPU devices. `nvidia-smi` command will be helpful if you don't know which device are available.
6. Run `python test.py` to evaluate your model on the test set.

## Important notice for CUDA 9 users
If you use >= CUDA 9, make sure you set tensorflow_gpu >= 1.5.

## Author
Bruce Chou (brucechou1983@gmail.com)

## Editor
Ali Gholami (ali.gholami@sharif.edu)

## License
MIT
