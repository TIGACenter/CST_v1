import os

import json
import argparse
import tensorflow as tf

import CST as CST
import utils
from metrics import recall_m, precision_m, f1_m, auc_m

DEFAULT_CONFIG = {
    "alpha": 1.0,
    "dist_params": {
        "contrast": {"lower": 0.8, "upper": 1.2},
        "color": {"factor": [20,0,20]},
        "blur": {"kernel_size": 1, "sigma": 3.0},
        "brightness": {"max_delta":0.3}
    },
    "optimizer": {
        "class_name": "Adam",
        "config": {
            "lr": 0.0001,
            "amsgrad": True
        }
    },
    "loss": "binary_crossentropy",
    "class_mode": "binary",
    "pretrained_model_path": None,
    "save_all_epochs": True,
    "model_save_path": "../models",
    "model_name": "model_name",
    "save_metrics": True,
    "epochs": 10,
    "metrics": ["binary_crossentropy", "recall", "precision", "f1", "auc"],
    "train_data_path": "../data/histo",
    "val_data_path": None
}

DEFAULT_METRICS = {
    "recall": recall_m,
    "precision": precision_m,
    "f1": f1_m,
    "auc": auc_m
}


def train(config_file=None):
    # load json config file
    conf = DEFAULT_CONFIG
    if config_file is not None:
        with open(config_file) as c:
            conf = json.load(c)

    # create generator
    if conf["val_data_path"] is None:
        t_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=utils.normalize_image,
            validation_split=conf["val_split"]
        )
        t_gen.validation_split = conf["val_split"],
        t_flow = t_gen.flow_from_directory(
            directory=conf["train_data_path"],
            target_size=(conf["tile_size"], conf["tile_size"]),
            color_mode="rgb",
            batch_size=conf["batch_size"],
            class_mode=conf["class_mode"],
            subset="training",
            shuffle=True
        )
        v_flow = t_gen.flow_from_directory(
            directory=conf["train_data_path"],
            target_size=(conf["tile_size"], conf["tile_size"]),
            color_mode="rgb",
            batch_size=conf["batch_size"],
            class_mode=conf["class_mode"],
            shuffle=True,
            subset="validation"
        )
    else:
        t_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=utils.normalize_image
        )
        t_flow = t_gen.flow_from_directory(
            directory=conf["train_data_path"],
            target_size=(conf["tile_size"], conf["tile_size"]),
            color_mode="rgb",
            batch_size=conf["batch_size"],
            class_mode=conf["class_mode"],
            shuffle=True
        )
        v_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=utils.normalize_image
        )
        v_flow = v_gen.flow_from_directory(
            directory=conf["val_data_path"],
            target_size=(conf["tile_size"], conf["tile_size"]),
            color_mode="rgb",
            batch_size=conf["batch_size"],
            class_mode=conf["class_mode"],
            shuffle=True,
        )

    # load/create model
    output_node_size = 1 if conf["class_mode"] == "binary" else t_flow.num_classes
    model = utils.create_thesis_model(conf["tile_size"], 3, conf["pretrained_model_path"],
                                      output_node_size, "resnet")

    # create cst instance
    cst = CST.ContrastiveStabilityTraining(
        model=model,
        tile_size=conf["tile_size"],
        dist_params=conf["dist_params"],
        alpha=conf["alpha"]
    )

    # compile
    opt = tf.keras.optimizers.get(conf["optimizer"])
    metrics =[DEFAULT_METRICS[i] if i in DEFAULT_METRICS else i for i in conf["metrics"] ]
    loss = tf.keras.losses.get(conf["loss"])
    cst.compile_cst(optimizer=opt, metrics=metrics, loss=loss)

    # train
    class_weight = utils.get_class_weights(t_flow.classes)
    cst.train_cst(
        x=t_flow,
        validation_data=v_flow,
        save_all_epochs=conf["save_all_epochs"],
        model_save_path=conf["model_save_path"],
        model_name=conf["model_name"],
        save_metrics=conf["save_metrics"],
        class_weight=class_weight,
        epochs=conf["epochs"]
    )

    # savemodel
    cst.save_model(conf["model_save_path"], conf["model_name"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", help="Path of the json config file", type=str)
    args = parser.parse_args()

    train(config_file=args.conf)
