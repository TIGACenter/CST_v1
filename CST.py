import os

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
            self.model.layers[1].save(
                os.path.join(self.model_path, self.model_name) + "_e" + str(epoch + 1) + ".h5")
            print("class weights saved to path: ")
            print(self.model_name + str(epoch + 1) + ".h5")


class ContrastiveStabilityTraining:
    def __init__(self, model, tile_size, dist_params={}, alpha=0):
        self.model = model
        self.tile_size = tile_size
        self.dist_params = dist_params
        self.alpha = alpha
        self.build_cst_wrap()

    def build_cst_wrap(self):
        self.i = tf.keras.layers.Input(shape=(self.tile_size, self.tile_size, 3))
        self.i_dist = tf.keras.layers.Lambda(dist_fn,
            arguments={"dist_params": self.dist_params, "tile_size":self.tile_size})(self.i)

        self.x_i = self.model(self.i)
        self.x_i_dist = self.model(self.i_dist)
        self.cst_model = tf.keras.models.Model(inputs=self.i, outputs=self.x_i)

    def compile_cst(self, optimizer, loss, metrics=[]):
        self.cst_model.compile(
            optimizer=optimizer,
            loss=self.cst_loss(self.x_i_dist, self.alpha, loss),
            metrics=["acc", self.cst_loss(self.x_i_dist, self.alpha, loss)] + metrics
        )

    def train_cst(self, x, y=None, validation_data=None, save_all_epochs=False, model_save_path="",
                  model_name="model", save_metrics=False, class_weight={0: 1., 1: 1.}, workers=8,
                  use_multiprocessing=True, epochs=1, callbacks=[]):
        if save_all_epochs:
            callbacks.append(EpochSaver(model_save_path, model_name))
        if save_metrics:
            callbacks.append(tf.keras.callbacks.CSVLogger(
                os.path.join(model_save_path, model_name) + ".csv", ","))

        self.cst_model.fit(
            x=x,
            y=y,
            validation_data=validation_data,
            callbacks=callbacks,
            epochs=epochs,
            use_multiprocessing=use_multiprocessing,
            workers=workers,
            class_weight=class_weight
        )

    def save_model(self, model_save_path, model_name):
        self.cst_model.layers[1].save(os.path.join(model_save_path, model_name) + ".h5")

    @staticmethod
    def cst_loss(x_i_dist, alpha=1., l_0=tf.keras.losses.binary_crossentropy):
        def loss(y_true, y_pred):
            # l_0 = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            l_stability_1_col_1 = tf.keras.losses.kullback_leibler_divergence(y_pred, x_i_dist)
            l_stability_1_col_2 = tf.keras.losses.kullback_leibler_divergence(1-y_pred, 1-x_i_dist)
            l_stability = l_stability_1_col_1 + l_stability_1_col_2
            return l_0(y_true, y_pred)  + l_stability * alpha
        return loss


def dist_fn(images, tile_size=128, dist_params={}):
    """
    Images have to be normalized centered in 0
    """

    X = images
    if "color" in dist_params:
        pix_dist = dist_params["color"]["factor"]
        norm_pix_dist = [i * 2 / 255. for i in pix_dist]

        X_B = tf.squeeze(tf.slice(X, [0, 0, 0, 0], [-1, -1, -1, 1]))
        X_G = tf.squeeze(tf.slice(X, [0, 0, 0, 1], [-1, -1, -1, 1]))
        X_R = tf.squeeze(tf.slice(X, [0, 0, 0, 2], [-1, -1, -1, 1]))

        X_B = X_B + K.random_uniform(shape=[], minval=-norm_pix_dist[0],
                                     maxval=norm_pix_dist[0], dtype=tf.float32)
        X_G = X_G + K.random_uniform(shape=[],minval=-norm_pix_dist[1],
                                     maxval=norm_pix_dist[1], dtype=tf.float32)
        X_R = X_R + K.random_uniform(shape=[], minval=-norm_pix_dist[2],
                                     maxval=norm_pix_dist[2], dtype=tf.float32)

        X = tf.stack([X_B, X_G, X_R], axis=3)

    if "contrast" in dist_params:
        lower = dist_params["contrast"]["lower"]
        upper = dist_params["contrast"]["upper"]
        X = tf.image.random_contrast(X, lower, upper)

    if "brightness" in dist_params:
        max_delta = dist_params["brightness"]["max_delta"]
        X = tf.image.random_brightness(X, max_delta)

    if "blur" in dist_params:
        kernel_size = dist_params["blur"]["kernel_size"]
        sigma = dist_params["blur"]["sigma"]
        gauss_kernel = gaussian_kernel(size=kernel_size, mean=0.,
                                       std=np.random.random() * sigma + .00001)
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

    return tf.clip_by_value(X, -1, 1)









