import os
import pathlib

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from utils import gaussian_kernel


class EpochSaver(tf.keras.callbacks.Callback):
    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.model_name = model_name
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % 1 == 0:
            pathlib.Path(self.model_path).mkdir(parents=True, exist_ok=True)
            self.model.layers[-1].save(
                os.path.join(self.model_path, self.model_name) + "_e" + str(epoch + 1) + ".h5")
            print("class weights saved to path: ")
            print(self.model_name + str(epoch + 1) + ".h5")


class CNNStabilityTraining:
    def __init__(self, model, tile_size, dist_params={}, alpha=0, loss_modality="cst_loss"):
        self.model = model
        self.class_mode = "binary" if self.model.layers[-1].units == 1 else ""
        self.tile_size = tile_size
        self.dist_params = dist_params
        self.alpha = alpha
        self.loss_modality = loss_modality
        self.build_cst_wrap()

    # TODO: correctly include the cst vs the normal or the data augmented training modalities
    def build_cst_wrap(self):
        self.i = tf.keras.layers.Input(shape=(self.tile_size, self.tile_size, 3))
        self.i_dist = tf.keras.layers.Lambda(dist_fn,
            arguments={"dist_params": self.dist_params, "tile_size":self.tile_size})(self.i)

        print(f"Loss modality: {self.loss_modality}")

        if self.loss_modality == "da_loss":
            # train with da_loss
            self.i_norm = tf.keras.layers.Lambda(dist_fn,
                arguments={"dist_params": self.dist_params, "tile_size": self.tile_size})(self.i)
            self.x_i = self.model(self.i_norm)
            self.x_i_dist = self.x_i

        else:
            # train with cst_loss
            self.i_norm = tf.keras.layers.Lambda(dist_fn,
                arguments={"dist_params": {"normalize": self.dist_params["normalize"]},
                           "tile_size":self.tile_size})(self.i)
            self.x_i = self.model(self.i_norm)
            self.x_i_dist = self.model(self.i_dist)

        self.cst_model = tf.keras.models.Model(inputs=self.i, outputs=self.x_i)

    def compile_cst(self, optimizer, loss, metrics=[]):
        if self.loss_modality=="da_loss":
            cst_loss = self.da_loss(self.x_i_dist, loss, self.alpha, self.class_mode)
        else:
            cst_loss = self.cst_loss(self.x_i_dist, loss, self.alpha, self.class_mode)
        self.cst_model.compile(
            optimizer=optimizer,
            loss=cst_loss,
            metrics=["acc", cst_loss] + metrics
        )
        # print(self.cst_model.summary())

    def train_cst(self, x, y=None, validation_data=None, save_all_epochs=False, model_save_path="",
                  model_name="model", save_metrics=False, class_weight={0: 1., 1: 1.}, workers=8,
                  use_multiprocessing=True, epochs=1, callbacks=None):

        model_new_dir = os.path.join(model_save_path, model_name)

        if callbacks is None:
            callbacks = []

        if save_all_epochs:
            pathlib.Path(model_new_dir).mkdir(parents=True, exist_ok=True)
            callbacks.append(EpochSaver(model_new_dir, model_name))
        if save_metrics:
            pathlib.Path(model_new_dir).mkdir(parents=True, exist_ok=True)
            callbacks.append(tf.keras.callbacks.CSVLogger(
                os.path.join(model_new_dir, model_name) + ".csv", ","))

        self.cst_model.fit(
            x=x,
            y=y,
            validation_data=validation_data,
            callbacks=callbacks,
            epochs=epochs,
            use_multiprocessing=use_multiprocessing,
            class_weight=class_weight,
            workers=workers,

        )

    def save_model(self, model_save_path, model_name):
        model_new_dir = os.path.join(model_save_path, model_name)
        pathlib.Path(model_new_dir).mkdir(parents=True, exist_ok=True)
        self.cst_model.layers[-1].save(os.path.join(model_new_dir, model_name) + ".h5")

    @staticmethod
    def cst_loss(x_i_dist, l_0, alpha, class_mode):
        """
        loss function for CST, wraps the loss in order to pass values that are not y_true y_pred,
        which is a limitation of this tf version
        """
        def internal_loss(y_true, y_pred):
            l = l_0(y_true, y_pred)
            l_stability = tf.keras.losses.kullback_leibler_divergence(y_pred, x_i_dist)
            if class_mode == "binary":
                l_stability = l_stability + \
                              tf.keras.losses.kullback_leibler_divergence(1-y_pred, 1-x_i_dist)
            return l + l_stability * alpha
        return internal_loss

    @staticmethod
    def da_loss(x_i_dist, l_0, alpha=0, class_mode=None):


        def internal_loss(y_true, y_pred):
            l = l_0(y_true, y_pred)
            return l

        return internal_loss

# TODO: fix normalization bug, as contrast and brightness require RGB images
def dist_fn(images, tile_size=128, dist_params={}):
    """
    Images have to be normalized and centered in 0
    """
    X = images
    if "color" in dist_params:
        pix_dist = dist_params["color"]["factor"]

        X_B = tf.squeeze(tf.slice(X, [0, 0, 0, 0], [-1, -1, -1, 1]))
        X_G = tf.squeeze(tf.slice(X, [0, 0, 0, 1], [-1, -1, -1, 1]))
        X_R = tf.squeeze(tf.slice(X, [0, 0, 0, 2], [-1, -1, -1, 1]))

        X_B = X_B + K.random_uniform(shape=[], minval=-pix_dist[0],
                                     maxval=pix_dist[0], dtype=tf.float32)
        X_G = X_G + K.random_uniform(shape=[],minval=-pix_dist[1],
                                     maxval=pix_dist[1], dtype=tf.float32)
        X_R = X_R + K.random_uniform(shape=[], minval=-pix_dist[2],
                                     maxval=pix_dist[2], dtype=tf.float32)

        X = tf.stack([X_B, X_G, X_R], axis=3)

    X = tf_normalize(X)

    if "contrast" in dist_params:
        lower = dist_params["contrast"]["lower"]
        upper = dist_params["contrast"]["upper"]
        X = tf.image.random_contrast(X, lower, upper)

    if "brightness" in dist_params:
        max_delta = dist_params["brightness"]["max_delta"]
        X = tf.image.random_brightness(X, max_delta)

    if "blur" in dist_params and dist_params["blur"] is not None:
        kernel_size = dist_params["blur"]["kernel_size"]
        sigma = dist_params["blur"]["sigma"]
        gauss_kernel = gaussian_kernel(size=kernel_size, mean=0.,
                                       # std=np.random.random() * sigma / 255. + .00001)
                                       std = np.random.random() * sigma + .00001)
        gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
        X_B = tf.slice(X, [0, 0, 0, 0], [-1, -1, -1, 1])
        X_G = tf.slice(X, [0, 0, 0, 1], [-1, -1, -1, 1])
        X_R = tf.slice(X, [0, 0, 0, 2], [-1, -1, -1, 1])
        X_B = tf.nn.conv2d(X_B, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")
        X_G = tf.nn.conv2d(X_G, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")
        X_R = tf.nn.conv2d(X_R, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")
        X_B = tf.squeeze(X_B)
        X_G = tf.squeeze(X_G)
        X_R = tf.squeeze(X_R)
        X = tf.stack([X_B, X_G, X_R], axis=3)
        X = tf.image.resize_bilinear(X, tf.constant([tile_size, tile_size]))


    if "normalize" not in dist_params or dist_params["normalize"] is False:
        X = tf_denormalize(X)
        return tf.clip_by_value(X, 0, 255)

    return tf.clip_by_value(X, -1, 1)

def tf_normalize(imgs):
    imgs = tf.multiply(tf.cast(imgs, tf.float32), 1 / 255)
    imgs = tf.add(imgs, -0.5)
    return tf.multiply(imgs, 2)

def tf_denormalize(imgs):
    imgs = tf.multiply(imgs, 1 / 2)
    imgs = tf.add(imgs, 0.5)
    return tf.multiply(imgs, 255)