import os
import shutil
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve
from math import floor
from cv2 import cvtColor, COLOR_BGR2HSV, normalize, CV_32F, NORM_MINMAX, blur, calcHist, Laplacian, COLOR_RGB2GRAY, CV_64F
from collections import Counter
import tensorflow as tf


def create_directory(*args):
    '''
    *args are strings corresponding to directories
    '''
    for directory in args:
        os.mkdir(os.path.dirname(directory))


def build_dirs(directory):
    try:
        shutil.rmtree("split/X/")
    except:
        pass

    create_directory("split/X/")
    create_directory("split/X/-1/",  # background
                     "split/X/1/",  # epithelium
                     "split/X/0/")  # non-epithelium


def list_files_from_dir(directory="", extension=[".ndpi"]):
    # TODO> pasar a "utils.py"
    '''
    lists files of extension <extension> in directory.
    It also returns the path relative to the inputed directory
    '''
    # TODO> resolver el bug... cuando corro preprocess o split la variable glb
    #       tiene que tener "**/*" pero para q funcione con train.py y test.py tiene
    #       tiene que tener "/**/*"

    glb = [glob.glob(directory + "/**/*" + e, recursive=True) for e in extension]
    glb = [i for sublist in glb for i in sublist]

    file_list = [os.path.basename(f) for f in glb]
    dir_list = [os.path.dirname(f).replace(directory + "\\", "") for f in glb]
    counts = dict(Counter(dir_list))

    return file_list, dir_list, counts


def save_np_as_image(np_array, filename):
    im = Image.fromarray(np.uint8(np_array))
    im.save(filename)
    return


def extract_region(np_array, square_top_left_corner, square_height,
                   square_width, draw_region=False):

    '''
    square_top_left_corner: (x,y) tuple
    square_height and square_width: height and width of ROI
    draw_region: if true, the extracted ROI will be previewed
    '''
    region = np_array[square_top_left_corner[1]:
                      square_top_left_corner[1]+square_height,
                      square_top_left_corner[0]:
                      square_top_left_corner[0]+square_width]

    if draw_region:
        plt.figure()
        plt.imshow(region*255, aspect='auto')
        plt.show()

    return region


def to_hsv(image):
    '''
    Image corresponds to a numpy array.
    function equires original image to come in RGB.
    NDPI images extracted with Openslide come in RGBA,
    so the last channel is dismissed.
    '''
    return cvtColor(image, COLOR_BGR2HSV)


def normalize_image(image, hsv=False):
    '''
    image corresponds to a numpy array.
    '''
    
    norm_image = image / 255.
    
    if hsv:
        norm_image[:,:,0] = norm_image[:,:,0] * 255. / 179.
    
    norm_image -= 0.5
    norm_image *= 2.

    return norm_image


def blur_image(image, kernel_shape=(4,4)):
    kernel = np.ones((kernel_shape[0],kernel_shape[1], 3)) / (kernel_shape[0]*kernel_shape[1]*3)
    return convolve(image, kernel)


def train_validation_test_partition(file_list, prop=(0.6, 0.4, 0.0)):
    lf = len(file_list)
    indexes = np.arange(lf)
    np.random.seed(9)
    np.random.shuffle(indexes)
    train_list = [file_list[indexes[i]]
                  for i in range(0, floor(prop[0]*lf))]
    val_list = [file_list[indexes[i]]
                for i in range(floor(prop[0]*lf),
                               floor((prop[0]+prop[1])*lf))]
    test_list = [file_list[indexes[i]]
                 for i in range(floor((prop[0]+prop[1])*lf),
                                floor((prop[0]+prop[1]+prop[2])*lf))]
    return train_list, val_list, test_list


def gaussian_kernel(size: int, mean: float, std: float):
    """Makes 2D gaussian Kernel for convolution."""
    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    gauss_kernel = gauss_kernel/tf.reduce_sum(gauss_kernel)
    return gauss_kernel


def tile_is_background_1(image, threshold=0.85):
    '''
    returns True if tile (np array) is background. An <image> is classified
    as background if the proportion of pixel colors (gray scale, 0-255)
    within a range of +-10 from the histogram mode is over <threshold>.

    inputs:
     - image: numpy array corresponding to RGB image
     - threshold: if (pixels in rng / total pixels)
                is higher than threshold, images is classified as background
    '''
    is_bkgnd = False
    hist = calcHist(images=[cvtColor(image, COLOR_RGB2GRAY)],
                    channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    mode = np.argmax(hist)
    if np.sum(hist[mode-10:mode+10])/np.sum(hist) > threshold:
        is_bkgnd = True
    
    abs_lap = np.absolute(Laplacian(cvtColor(image, COLOR_RGB2GRAY), CV_64F))
    if abs_lap.max() < 40:
        is_bkgnd = True
    return hist, is_bkgnd