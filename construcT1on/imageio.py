''' Utilities for saving and reading niftis and pngs '''
from __future__ import division

import scipy.misc
import numpy as np

import nibabel as nb
from nilearn import plotting

def get_nifti_image(image_path, image_size):
    ''' Given '.nii' file, return an arbitrary 2D slice, shape (image_size, image_size, 1) '''

    def get_pad_width(dim):
        ''' how much to pad the array on each size to achieve image_size '''
        assert dim <= image_size
        diff = image_size - dim
        pad = diff // 2

        if np.mod(diff, 2) == 1:
            return pad, pad + 1
        return pad, pad

    image_array = nb.load(image_path).get_data()[:, :, 50]
    assert image_array.ndim == 2
    length, width = image_array.shape

    length_pads = get_pad_width(length)
    width_pads = get_pad_width(width)

    image_array = np.pad(image_array, (length_pads, width_pads), 'constant', constant_values=0)
    image_array = as_3d(fit_range(image_array))

    save_nifti_image(image_array, '/home/berleant/trainingimg.png')
    return image_array

def fit_range(array):
    ''' for whatever reason this seems to work better with a range from -1 to 1
    than the native values from <0 to 99968 found in the MNI and HCP T1ws'''
    pre_max = np.amax(array)
    pre_min = np.amin(array)
    midpoint = (pre_max + pre_min) / 2.

    array = (array - midpoint)/(pre_max - midpoint)

    post_max = np.amax(array)
    post_min = np.amin(array)
    epsilon = 0.01
    assert np.absolute(1 - post_max) < epsilon
    assert np.absolute(1 + post_min) < epsilon

    return array

def as_3d(image_array):
    ''' converts 2d array to a a 3d array with the last axis being size 1 '''
    if image_array.ndim == 2: # we want a 3-D array
        image_array = image_array[:, :, np.newaxis]
    return image_array

def save_nifti_images(matrices, size, image_path):
    ''' Given 4d numpy array of matrices, save a summary png '''
    assert matrices.ndim == 4
    images = None
    for matrix in matrices:
        image = save_nifti_image(matrix)
        images = image if images is None else np.concatenate((images, image))
    save_images(images, size, image_path)

def save_nifti_image(matrix, png_path='tmp.png'):
    ''' given a 3d numpy array representing a slice of a brain, save a summary png and return scipy
    representation. hacky if png is not actually needed (most cases) '''
    # think about affine
    nifti = nb.Nifti1Image(as_3d(matrix), np.eye(4))
    plotting.plot_anat(
        nifti, output_file=png_path, draw_cross=False, annotate=False, display_mode='z')
    return imread(png_path, mode='L')[np.newaxis, :, :]

def save_images(images, size, image_path):
    ''' saves the list of images in a size[0] by size[1] grid at image_path '''
    return imsave(inverse_transform(images), size, image_path)

def imread(path, mode):
    ''' just a wrapper around scipy.misc.imread '''
    return scipy.misc.imread(path, mode=mode).astype(np.float)

def merge(images, size):
    ''' makes an image out of images in a size[0] by size[1] grid '''
    height, width = images.shape[1], images.shape[2]
    img = np.zeros((int(height * size[0]), int(width * size[1])))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * height:j * height + height, i * width:i * width + width] = image
    return img

def imsave(images, size, path):
    ''' just a wrapper around scipy.misc.imsave '''
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(images):
    ''' prepares the numpy array to be saved as a png '''
    return (images+1.)/2.
