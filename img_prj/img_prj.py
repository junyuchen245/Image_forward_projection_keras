"""
Author: Junyu Chen
Date: 3/10/2020

Image forward projection layer in Keras/TensorFlow
"""

import keras.layers as KL
from keras.layers import *
from keras.models import Model, load_model
import numpy as np
import scipy.stats as st
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from keras import backend as K
from PIL import Image
from skimage.transform import rescale, resize


class image_prj(Layer):
    """
    Image projection layer.
    prj_angles: a list of angles
    """

    def __init__(self, prj_angles=(0,180,270), **kwargs):
        self.prj_angles = prj_angles
        super(image_prj, self).__init__(**kwargs)

    def build(self, input_shape):
        super(image_prj, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        prj_out = None
        for i in range(0,len(self.prj_angles)):
            # extract an angle
            angle = self.prj_angles[i]*np.pi/180
            # rotate image by angle
            imgRot= tf.contrib.image.rotate(x, angle, interpolation='NEAREST')
            # sum image in one direction
            if prj_out == None:
                prj_out = tf.reduce_sum(imgRot, axis=1, keepdims=True)
            else:
                prj_out = tf.concat((prj_out, tf.reduce_sum(imgRot, axis=1, keepdims=True)), axis=1)
        return prj_out


"""
********* Test "image_prj" layer **********
"""

def prj(input_size = (192,192,1), angles=(0, 1, 2, 3, 4, 5, 6, 7)):
    img_in = Input(input_size)
    prj_out = image_prj(prj_angles=angles)(img_in)

    model = Model(inputs=img_in, outputs=prj_out)
    return model

if K.backend() == 'tensorflow':
    # Use only gpu #X (with tf.device(/gpu:X) does not work)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # Automatically choose an existing and supported device if the specified one does not exist
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # To constrain the use of gpu memory, otherwise all memory is used
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    print('GPU Setup done')

sz_x, sz_y = 192,192

#img = np.load('/netscratch/jchen/CNNClustering/u01_pat_sim.npz')

#img = img['arr_0']
#img = img[200,:,:].reshape(1, sz_x, sz_y, 1)
img = np.array(Image.open('SheppLogan_Phantom.png').convert('LA'))[:,:,0]
img = resize(img, (sz_x, sz_y), anti_aliasing=False)
img = img.reshape(1, sz_x, sz_y, 1)
angles = range(0,180,1)
prj_model = prj((sz_x,sz_y,1),angles)
y = prj_model.predict(img)
# print(np.max(np.abs(y)))
plt.figure()
# plt.figure(num=None, figsize=(10, 10), dpi=100, facecolor='w', edgecolor='k')
plt.imshow(y[0, :, :, 0], cmap = 'gray')
plt.savefig('out.png')




