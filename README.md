# Convolutional Neural Network Stability Training

Convolutional Neural Network Stability Training (CST), in the context of deep learning for digital pathology, is proposed as a method to increase DL model robustness to image variability caused by the use of different slide scanners and IHC stains. Distortions in color, contrast, brightness and blur were used for CST, in accordance with the perceived differences between WSIs  from different scanners and with different IHC.

<img align="center" src="static/cst_segmentation.png" width="800">

## Notes
Tested with Nvidia Geforce RTX 2070 and 2080. May not work with RTX 3000 series.

## Requirements

python 3.6

CUDA 10.0 (or CUDA >= 10.0 for the docker setup)

python libraries: 
- protobuf<4.21.0
- tensorflow-gpu==1.13.1
- numpy==1.16.4
- scipy==1.2.1
- opencv-python==4.1.0.25
- matplotlib
- ipykernel
- pillow==6.0.0
- h5py==2.10.0
- requests==2.27.1

## Setup

#### With docker 

Build image:

``` bash
$ docker build -f Dockerfile --tag cst_v1 .
```

Run image:

*OBS: If you need to run docker with sudo, make sure to create the `</path/to/data>`, `</path/to/config>` and `</path/to/models>` folders beforehand. Otherwise, `sudo docker run ...` will create the folders with root permissions, and scripts within docker will encounter permission issues.

```bash
$ docker run -p <port> \
             --network="host" \
             --user $(id -u):$(id -g) \
             -it --rm \
             -v </path/to/data>:/main_dir/CST_v1/data/ \
             -v </path/to/config>:/main_dir/CST_v1/config/ \
             -v </path/to/models>:/main_dir/CST_v1/models/ \
             --gpus <gpus> 
             cst_v1:latest /bin/bash
# e.g. docker run -p 8088:8088 \
#                 --network="host" \
#                 --user $(id -u):$(id -g)  \
#                 -it --rm \
#                 -v ~/local_cst/data/:/main_dir/CST_v1/data/ \
#                 -v ~/local_cst/config/:/main_dir/CST_v1/config/ \
#                 -v ~/local_cst/models/:/main_dir/CST_v1/models/ \
#                 --gpus all \
#                 cst_v1:latest /bin/bash
# (change port, select gpus, add user permissions (e.g. --user 1000:1000) or generally modify the run command as needed.)

```

#### Without docker 

*OBS: Might not be that easy to install cuda + nvidia driver + cudnn versions compatible with tf 1.13. Version compatibility can be found [here](https://www.tensorflow.org/install/source#gpu).

1. Install cuda...

Download the CUDA Toolkit [here](https://developer.nvidia.com/cuda-downloads?target_os=Linux).

2. Install libraries: 
```bash 
$ pip install -r requirements.txt
```



## Usage

#### With docker:

Download datasets: 
```bash
$ python3 load_data.py --name <dataset1> <dataset2> <...> \
                       --path <path>
  
  
# options for --name are idc, cifar-10, cifar-10-c. Defaults to idc.
# path corresponds to the download path. Defaults to /main_dir/CST_v1/models/
            
# e.g. $ python3 load_data.py --name idc cifar-10
```

Train network:

```bash
$ python3 train.py --conf <conf_path> # conf_path is the path to a config json file with the parameters. Default is train_config.json
# e.g. $ python3 train.py --conf train_config.json
```

Run jupyter notebooks:
```bash
$ jupyter notebook
```

Then open the url in your browser.

#### Without docker:

Same us above. If python 3.6 is your default version, just use python instead of python3.6


## Config file for training
This file contains all the parameters that need to be set to train a network using CST. The file "train_config.json" is included for illustration purposes:

```jsonc
{
    "tile_size": 128,                             // size of the image (for now only square image)
    "alpha": 1.0,                                 // weight of stability component
    "dist_params": {                              // dict containing the distortions
        "contrast": {"lower": 0.8, "upper": 1.2}, // random contrast within range [lower, upper]
        "color": {"factor": [20,0,20]},           // random color on each RGB channel within [-factor[i], factor[i]] (0 to 255)
        "blur": {"kernel_size": 1, "sigma": 3.0}, // conv with gaussian kernel of size 2*kernel_size+1 and random std within [0, sigma]
        "brightness": {"max_delta":0.3},          // random brightness within [-delta, delta]
        "normalize": true                         // normalizes to range [-1, 1] (i.e. ( (x / 255.) - 0.5) * 2 )
    },
    "optimizer": {                                // optimizer (tf.keras.optimizers)
        "class_name": "Adam",                     // identifier (e.g. "Adam", "SGD", ...). opt is called by tf.keras.optimiters.get(identifier)
        "config": {                               // args for, in this case, tf.keras.optimizers.Adam
            "lr": 0.0001,
            "amsgrad": true
        }
    },
    "loss": "binary_crossentropy",                // loss (tf.keras.losses)
    "loss_modality": "cst_loss",                  // "cst_loss" applies CNN stability training, "da_loss" applies traditional on-the-fly data augmentation with the predefined dist_params
    "class_mode": "binary",                       // class mode ("binary" for 2 classes or "categorical" for 3+)
    "pretrained_model_path": null,                // path of custom model if exists. CST will be used to retrain
    "save_all_epochs": true,                      // if true, all epochs are saved
    "model_save_path": "models",                  // path to save models
    "model_name": "default_model_name",           // name of models to save, will be saved as "model_name.h5"
    "save_metrics": true,                         // if true, creates csv and adds metrics on epoch end
    "epochs": 10,                                 // n of epochs
    "batch_size": 32,                             // batch size
    "metrics": ["binary_crossentropy", "recall_m", "precision_m", "f1_m", "auc_m"],  // metrics (retrieved with tf.keras.metrics.get()) Additionaly, "recall_m", "precision_m", "f1_m", "auc_m" are custom metrics available optionally
    "train_data_path": "data/aj/IDC_regular_ps50_idx5",  // path of train data
    "val_data_path": null,                        // path of validation data
    "val_split": 0.2,                             // validation split if "val_data_path" is null
    "arq": "InceptionV3",                         // used if pretrained_model_path is null. Only "resnet" (tf.keras.models.ResNet50) and "InceptionV3" (tf.keras.Models.InceptionV3)
    "classes": ["0", "1"]                         // classes to be used for training. If null, he list of classes will be automatically inferred from the train_data_paths subdirectories
}
```



## License
[MIT](https://choosealicense.com/licenses/mit/)

## Reference

This is the [link](https://www.frontiersin.org/articles/10.3389/fmed.2023.1173616/full) to our paper.

Miranda Ruiz F, Lahrmann B, Bartels L, Krauthoff A, Keil A, Härtel S, Tao AS, Ströbel P, Clarke MA, Wentzensen N and Grabe N (2023) CNN stability training improves robustness to scanner and IHC-based image variability for epithelium segmentation in cervical histology. Front. Med. 10:1173616. doi: 10.3389/fmed.2023.1173616

```
@article{miranda10cnn,
  title={CNN Stability Training improves robustness to scanner and IHC-based image variability for epithelium segmentation in cervical histology},
  author={Miranda Ruiz, Felipe and Lahrmann, Bernd and Bartels, Liam and Krauthoff, Alexandra and Keil, Andreas and Tao, Amy S and Stroebel, Philipp and Clarke, Megan A and H{\"a}rtel, Steffen and Wentzensen, Nicolas and others},
  journal={Frontiers in Medicine},
  volume={10},
  pages={1173616},
  publisher={Frontiers}
}
```