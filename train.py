# -*- coding: utf-8 -*-
import os

from utils import list_files_from_dir, normalize_image, \
    train_validation_test_partition, gaussian_kernel
from metrics import recall_m, precision_m, f1_m, auc_m
import cv2
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
import time
import json


class ImageGenerator2(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, image_label_directory, source_directory, tile_side=128,
                 batch_size=100, shuffle=True):
        self.source_directory = source_directory
        self.list_IDs = list_IDs
        self.image_label_directory = image_label_directory
        self.tile_side = tile_side
        self.batch_size = batch_size
        self.shuffle = shuffle
        np.random.seed(24)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))
        # return len(self.list_IDs)

    def __getitem__(self, i):
        # generate indexes of the batch
        indexes = self.indexes[i*self.batch_size: (i+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        '''
        generates batches of shape:
        (n_samples, tile_side, tile_side, n_channels)

        input:
         - list_IDs_temp: list with image filenames. For now, it only consists
           on a list with one element
        '''

        X = np.empty((self.batch_size, self.tile_side, self.tile_side, 3), dtype='float32')
        y = np.empty((self.batch_size), dtype=int)

        for i, fname in enumerate(list_IDs_temp):
            label = os.path.basename(self.image_label_directory[fname])
            X[i] = cv2.imread(self.source_directory + "/" + label + "/" + fname)
            X[i] = normalize_image(image=X[i])
            y[i] = int(label.replace("-1", "0"))
        return X, y


class CustomSaver(tf.keras.callbacks.Callback):
    def __init__(self, model_name, tile_side):
        self.model_name = model_name
        self.tile_side = tile_side
        super().__init__()
    def on_epoch_end(self, epoch):
        if (epoch + 1) % 1 == 0:
            self.model.layers[1].save(self.model_name.format(self.tile_side, epoch+1))
            print("class weights saved to path: ")
            print(self.model_name.format(self.tile_side, epoch+1))


def dist_fn(norm_image_tensor, color=False, contrast=False, brightness=False,
                              blur=False, noise=False):

    tf_image = norm_image_tensor

    if color:
        pix_dist = [20,0,20]
        norm_pix_dist = [i * 2 / 255. for i in pix_dist]

        tf_image_B = tf.slice(tf_image,[0,0,0,0],[-1,-1,-1,1])
        tf_image_G = tf.slice(tf_image,[0,0,0,1],[-1,-1,-1,1])
        tf_image_R = tf.slice(tf_image,[0,0,0,2],[-1,-1,-1,1])

        tf_image_B = tf.squeeze(tf_image_B)
        tf_image_G = tf.squeeze(tf_image_G)
        tf_image_R = tf.squeeze(tf_image_R)

        tf_image_B = tf_image_B + K.random_uniform(shape=[],minval=-norm_pix_dist[0], maxval=norm_pix_dist[0], dtype=tf.float32)
        tf_image_G = tf_image_G + K.random_uniform(shape=[],minval=-norm_pix_dist[1], maxval=norm_pix_dist[1], dtype=tf.float32)
        tf_image_R = tf_image_R + K.random_uniform(shape=[],minval=-norm_pix_dist[2], maxval=norm_pix_dist[2], dtype=tf.float32)

        tf_image = tf.stack([tf_image_B, tf_image_G, tf_image_R], axis=3)

    if contrast:
        tf_image = tf.image.random_contrast(tf_image, .6, 1.6)
        # tf_image = tf.image.random_contrast(tf_image, .8, 1.2)

    if brightness:
        tf_image = tf.image.random_brightness(tf_image, .5)
        # tf_image = tf.image.random_brightness(tf_image, .3)

    if blur:
        gauss_kernel = gaussian_kernel(size=2, mean=0., std=np.random.random()*5+.00001)
        gauss_kernel = gauss_kernel[:,:,tf.newaxis, tf.newaxis]
        tf_image_B = tf.slice(tf_image, [0,0,0,0],[-1,-1,-1,1])
        tf_image_G = tf.slice(tf_image, [0,0,0,1],[-1,-1,-1,1])
        tf_image_R = tf.slice(tf_image, [0,0,0,2],[-1,-1,-1,1])
        tf_image_B = tf.nn.conv2d(tf_image_B, gauss_kernel, strides=[1,1,1,1], padding="VALID")
        tf_image_G = tf.nn.conv2d(tf_image_G, gauss_kernel, strides=[1,1,1,1], padding="VALID")
        tf_image_R = tf.nn.conv2d(tf_image_R, gauss_kernel, strides=[1,1,1,1], padding="VALID")
        tf_image_B = tf.squeeze(tf_image_B)
        tf_image_G = tf.squeeze(tf_image_G)
        tf_image_R = tf.squeeze(tf_image_R)
        tf_image = tf.stack([tf_image_B, tf_image_G, tf_image_R], axis=3)
        tf_image = tf.image.resize_bilinear(tf_image, tf.constant([128,128]))

    if noise:
        tf_image_B, tf_image_G, tf_image_R = tf.unstack(tf_image, axis=-1)
        gauss_1 = tf.random_normal(shape=tf.shape(tf_image_B), mean=0.0, stddev=0.001)
        gauss_2 = gauss_1 + tf.random_normal(shape=tf.shape(tf_image_B), mean=0.0, stddev=0.0001)
        gauss_3 = gauss_1 + tf.random_normal(shape=tf.shape(tf_image_B), mean=0.0, stddev=0.0001)

        tf_image_B = gauss_1
        tf_image_G = gauss_2
        tf_image_R = gauss_3

        tf_image = tf.stack([tf_image_R, tf_image_G, tf_image_B], axis=-1)

    return tf.clip_by_value(tf_image, -1, 1)


def custom_loss(i_dist_output_layer, i_output_layer, i, i_dist, alpha=1.):

    def loss(y_true, y_pred):
        l_0 = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        l_stability_1_col_1 = tf.keras.losses.kullback_leibler_divergence(y_pred, i_dist_output_layer)
        l_stability_1_col_2 = tf.keras.losses.kullback_leibler_divergence(1-y_pred, 1-i_dist_output_layer)
        l_stability = l_stability_1_col_1 + l_stability_1_col_2

        return l_0  + l_stability * alpha

    return loss

def basic_dl_model(tile_side, saver, model_name, training_generator, validation_generator=None,
                   class_weight={0: 1., 1: 1.}, epochs=5, base="basic", trainable=True, opt="SGD",
                   lr=0.001, decay=1e-6, momentum=0.9, alpha=1., dropout=0., from_pretrained=False,
                   pretrained_dir=None):

    if not from_pretrained:
        if base == "Xception":
            conv_base = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(tile_side,tile_side,3))
        elif base == "InceptionV3":
            conv_base = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(tile_side,tile_side,3))
        elif base == "InceptionResNetV2":
            conv_base = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(tile_side,tile_side,3))
        elif base == "ResNet50":
            conv_base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(tile_side,tile_side,3))
        else:
            conv_base = tf.keras.models.Sequential([
                    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), input_shape=(tile_side, tile_side, 3), data_format="channels_last", activation='relu'),
                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), data_format="channels_last", activation='relu'),
                    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), data_format="channels_last", activation='relu'),
                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), data_format="channels_last", activation='relu'),
                    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), data_format="channels_last", activation='relu'),
                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2))])


        base_model = tf.keras.models.Sequential()
        base_model.add(conv_base)
        base_model.add(tf.keras.layers.Flatten())
        base_model.add(tf.keras.layers.Dense(128, activation='relu'))
        base_model.add(tf.keras.layers.Dense(64, activation='relu'))
        base_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    else:
        base_model = tf.keras.models.load_model(pretrained_dir,
            custom_objects={
                'precision_m': precision_m,
                'recall_m': recall_m,
                'f1_m': f1_m,
                'auc_m': auc_m,})


    i = tf.keras.layers.Input(shape=(tile_side, tile_side, 3))
    i_dist = tf.keras.layers.Lambda(dist_fn, arguments={'color': True, 'contrast': True, 'brightness': True, 'blur': True})(i)

    x_i = base_model(i)
    x_i_dist = base_model(i_dist)
    model = tf.keras.models.Model(inputs=i, outputs=x_i)


    '''compile/fit'''
    print(model.summary())

    if opt=="Adam":
        optimizer = tf.keras.optimizers.Adam(lr=lr, decay=decay, amsgrad=True)
    else:
        optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay)

    model.compile(optimizer=optimizer,
                  loss=custom_loss(x_i_dist, x_i, i, i_dist, alpha=alpha),
                #   loss='binary_crossentropy',
                  metrics=['acc', precision_m, recall_m, f1_m, auc_m, 'binary_crossentropy', custom_loss(x_i_dist, x_i, i, i_dist, alpha=alpha)])

    # $ tensorboard --logdir=./
    # tensorboard = tf.keras.callbacks.TensorBoard(log_dir='log/{}'.format(model_name.format(tile_side, "all")))

    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                #   callbacks=[saver, tensorboard],
                                  callbacks=[saver],
                                  epochs=epochs,
                                  use_multiprocessing=True,
                                  workers=8,
                                  class_weight=class_weight)
    return model, history


def main():
    '''
    trains a model from a set of tiles previously preprocessed and separated into
    folders according to the label
    '''
    ###  ["modeRange", "1", "InceptionV3", True, "BCE", "RGB", "p16", "15pics", "SGD", 0.0001, 1e-6, 0.9, 15, 1., True, 0.]
    ###  [     0        1         2          3     4      5      6        7       8       9     10    11  12  13   14   15]

    ### 0: type of background filterinf --> "stdDev" or "modeRange"
    ### 1: labelling of partially annotated triles --> "0" (non-epithelium) or "1" (epithelium)
    ### 2: DL base architecture --> "InceptionResNetV2", "InceptionV3", "ResNet50", "Xception", "basic" (which is default)
    ### 3: Variable that determines if layers of base architecture are trainable (no transfer learning) or not (transfer learning) --> True, False
    ### 4: Loss function --> "BCE" (binary crossentropy) or "ST" (stability training) (TODO: todavia no esta implementado este parametro, falta terminarlo)
    ### 5: Color model --> "RGB" or "HSV"
    ### 6: IHC used --> "p16" or "p16+CD8"
    ### 7: images used --> "15pics" or "Box1-4"
    ### 8: optimizer --> "Adam" o "SGD"
    ### 9: learning rate of optimizer --> 0.0001, 0.001, etc...
    ### 10: decay of optimizer --> 1e-6, 0.0001, 0.001, etc...
    ### 11: momentum of SGD optimizer (only useful if parameter 8 is "SGD") --> 0.9, 0.5, etc...
    ### 12: number of epochs
    ### 13: alpha for stability training --> has to be a number
    ### 14: Train with background --> True, False (TODO)
    ### 15: dropout
    ### 16: batch size

    parameter_sets = [
                      ["modeRange", "0", "InceptionV3", True, "ST", "RGB", "p16", "Box1-4", "SGD", 1e-3, 1e-6, 0.9, 20, 2., True, 0.0, 64], # 51 (es como el 43 pero con 30 epochs)
                    #   ["modeRange", "0", "InceptionV3", True, "ST", "RGB", "p16", "Box1-4", "SGD", 1e-3, 1e-6, 0.9, 20, 1., True, 0.0, 64],
                    #   ["modeRange", "0", "InceptionV3", True, "ST", "RGB", "p16", "Box1-4", "SGD", 1e-3, 1e-6, 0.9, 20, 2., True, 0.0, 64],
                    #   ["modeRange", "0", "InceptionV3", True, "ST", "RGB", "p16", "Box1-4", "SGD", 1e-3, 1e-6, 0.9, 20, 100., True, 0.0, 64],
                    #   ["modeRange", "0", "InceptionV3", True, "ST", "RGB", "p16", "15pics", "SGD", 1e-4, 1e-6, 0.9, 15, 1., True, 0.4, 64],
                    #   ["modeRange", "0", "InceptionV3", True, "ST", "RGB", "p16", "15pics", "SGD", 1e-4, 1e-6, 0.9, 15, 1., True, 0.2, 64],
                    #   ["modeRange", "0", "InceptionResNetV2", True, "ST", "RGB", "p16", "15pics", "SGD", 1e-4, 1e-6, 0.9, 15, 1., True, 0.2, 64],
                    #   ["modeRange", "0", "ResNet50", True, "ST", "RGB", "p16", "15pics", "SGD", 1e-4, 1e-6, 0.9, 15, 1., True, 0.2, 64],
                    #   ["modeRange", "0", "basic", True, "ST", "RGB", "p16", "15pics", "SGD", 1e-4, 1e-6, 0.9, 15, 1., True, 0.2, 64],
                    #   ["modeRange", "0", "InceptionV3", True, "ST", "RGB", "p16", "15pics", "Adam", 1e-4, 1e-5, 0.9, 15, 1., True, 0., 64], (default-Adam(22) + harta distortion)
                    #   ["modeRange", "0", "InceptionV3", True, "ST", "RGB", "p16+CD8", "Box1-4", "SGD", 1e-3, 1e-6, 0.9, 15, 10., True, 0.2, 64], # modelo SGD con los mejores parametros
                    #   ["modeRange", "0", "InceptionV3", True, "ST", "RGB", "p16", "15pics", "SGD", 1e-3, 1e-6, 0.9, 15, 10., True, 0.2, 64], # 40 (mejores parametros)
                    # ["modeRange", "0", "InceptionV3", True, "ST", "RGB", "p16", "Box1-4", "SGD", 1e-3, 1e-6, 0.9, 15, 1., True, 0.4, 64],
                    # ["modeRange", "0", "InceptionV3", True, "ST", "RGB", "p16", "Box1-4", "SGD", 1e-3, 1e-6, 0.9, 15, 1., True, 0.4, 64],
                      ]

    pretrained_dir = "models/20200606_InceptionV3_lossST1e+00_cRGB_Box1-4_modeRangePrep_pAre0_p16_optSGDlr1e-03d1e-06m9e-01_drop0.4_128px_e6-15.h5"

    for p_set in parameter_sets:


        name_base = "models/" + time.strftime("%Y%m%d") + "_" + p_set[2] + "_loss" + p_set[4] + \
            ("{:.0e}".format(p_set[13]) if p_set[4]=="ST" else "") + \
                "_c" + p_set[5] + "_" + p_set[7] + "_" + p_set[0] + "Prep_pAre" + \
                    p_set[1] + "_" + p_set[6] + "_opt" + p_set[8]

        if p_set[8] == "SGD":
            sgd_parameters = "lr{:.0e}d{:.0e}m{:.0e}".format(p_set[9], p_set[10], p_set[11])
        if p_set[8] == "Adam":
            sgd_parameters = "lr{:.0e}d{:.0e}".format(p_set[9], p_set[10])
        name_base = name_base + sgd_parameters

        bg_filter_dir = "128px_x20_RGB_{}_{}Preprocessing_partialsAre{}_{}/".format(p_set[7], p_set[0], p_set[1], p_set[6])
        # bg_filter_dir = "small_set/"
        full_dir = "/main_dir/felipe/schroederubuntu/projects/epi_seg/src/data/train/split/{}X".format(bg_filter_dir)
        # full_dir = "data/train/split/{}X".format(bg_filter_dir)
        file_list, dir_list, counts = list_files_from_dir(directory=full_dir,
                                                          extension=[".tif", ".png"])
        counts = {os.path.basename(i):v for i,v in counts.items()}

        if not p_set[14]:
            f_list = [file_list[i] for i in range(len(dir_list)) if dir_list[i] !="-1"]
            d_list = [dir_list[i] for i in range(len(dir_list)) if dir_list[i] !="-1"]
            file_list = f_list
            dir_list = d_list
            name_base = name_base + "_NoBkgd"

        name_base = name_base + "_drop" + str(p_set[15])

        model_name = name_base + "_{}px_e{}-" + str(p_set[12]) + ".h5"
        print(model_name)

        print("\n\nImage folder: " + bg_filter_dir)
        print("Image path: " + full_dir)
        print("Batch size: " + str(p_set[16]))
        print("Number of elements in file list: " + str(len(file_list)))
        print(str(counts) +"\n\n")
        print("From pretrained model: " + pretrained_dir)
        print("Base model: " + p_set[2])
        print("Base model trainable: " + str(p_set[3]))
        print("Optimizer: " + p_set[8] + " - " + sgd_parameters)
        print("Learning rate: " + str(p_set[9]))
        print("decay: " + str(p_set[10]))
        print("momentum: " + str(p_set[11]))
        print("Loss function: " + p_set[4])
        print("dropout: " + str(p_set[15]))
        if p_set[4]=="ST": print("alpha: " + str(p_set[13]))
        print("number of epochs: " + str(p_set[12]))

        train_list, val_list, _ = train_validation_test_partition(file_list, prop=(0.6, 0.4, 0.0))

        ild = {file_list[i]: dir_list[i] for i in range(len(dir_list))}

        tile_side = 128
        saver = CustomSaver(model_name=model_name, tile_side=tile_side)

        training_generator = ImageGenerator2(list_IDs=train_list,
                                             image_label_directory=ild,
                                             source_directory=full_dir,
                                             tile_side=tile_side,
                                             batch_size=p_set[16])
        validation_generator = ImageGenerator2(list_IDs=val_list,
                                               image_label_directory=ild,
                                               source_directory=full_dir,
                                               tile_side=tile_side,
                                               batch_size=p_set[16])

        class_weight = {0: 1., 1: (counts["-1"]+counts["0"])/counts["1"]}



        # epochs = [40]
        # for e in epochs:
        model, history = basic_dl_model(tile_side,
                                        saver=saver,
                                        model_name=model_name,
                                        training_generator=training_generator,
                                        validation_generator=validation_generator,
                                        class_weight=class_weight,
                                        epochs=p_set[12],
                                        base=p_set[2],
                                        trainable=p_set[3],
                                        opt=p_set[8],
                                        lr=p_set[9],
                                        decay=p_set[10],
                                        momentum=p_set[11],
                                        alpha=p_set[13],
                                        dropout=0.2,
                                        from_pretrained=True,
                                        pretrained_dir=pretrained_dir)

            # model.save(model_name.format(tile_side, e))

        history_name = name_base + "_" + str(tile_side) + "px" + str(p_set[12]) + "epochs.json"
        with open(history_name, 'w') as jsonfile:
            json.dump(history.history, jsonfile)

        K.clear_session()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
