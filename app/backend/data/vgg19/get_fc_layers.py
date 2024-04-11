import os
os.chdir("tensorflow-vgg-master")
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt
import requests
import pandas as pd
import numpy as np
import vgg19
import utils
import h5py
from tqdm import tqdm

# Load and transform the images into something inputtable to the model
# clothes_url = pd.read_csv().URL -- note: urls can be found in the data subdirectory of the main app

clothes_url1 = clothes_url[0:round((len(clothes_url)/2))]
clothes_url2 = clothes_url[round((len(clothes_url)/2)):len(clothes_url)]

batches1 = []
batches2 = []

# First batch of clothes images to preserve RAM
for url in tqdm(clothes_url1, desc="Loading First Batch"):
    batches1.append(utils.load_image(url).reshape((1, 224, 224, 3)))

np.savez("./first_batch_of_clothes.npz", batches1)

batches1 = []

# Second batch of clothes images
for url in tqdm(clothes_url1, desc="Loading Second Batch"):
    batches2.append(utils.load_image(url).reshape((1, 224, 224, 3)))

np.savez("./second_batch_of_clothes.npz", batches2)

batches2 = []

#################################################################################

mini_batch_size = 32  # mini batch size must be greater than 1


# Define VGG model output
def build_and_get_fc_layers(data, batch_size, output_file):
    # separate data into mini-batches
    split_batch = np.array_split(data, np.ceil(len(data) / batch_size))
    concat_array = np.array([]).reshape(0, 4096) # stores all the mini batch outputs
    for mini_batch in tqdm(split_batch):
        with tf.device('/gpu:0'):
            with tf.Session() as sess:
                # initialize model
                vgg = vgg19.Vgg19()

                images = tf.placeholder("float", [len(mini_batch), 224, 224, 3])
                feed_dict = {images: mini_batch}

                with tf.name_scope("content_vgg"):
                    vgg.build(images)

                # get layers and append it to the concat array
                relu7 = sess.run(vgg.relu7, feed_dict=feed_dict)
        concat_array = np.vstack([concat_array, relu7])
        tf.keras.backend.clear_session()

    np.savez(output_file, concat_array)


# First iteration

first_batch_ready = np.load("./first_batch_of_clothes.npz")

batch = np.concatenate((first_batch_ready['arr_0']), 0)
batch = np.reshape(batch, (407, 224, 224, 3))

# Run VGG model
build_and_get_fc_layers(batch, mini_batch_size, "./first_fc_layer.npz")

# Second iteration

second_batch_ready = np.load("./second_batch_of_clothes.npz")

batch = np.concatenate(second_batch_ready['arr_0'], 0)
batch = np.reshape(batch, (407, 224, 224, 3))

# Run VGG model
build_and_get_fc_layers(batch, mini_batch_size, "./second_fc_layer.npz")

# Load FCs
first_fc_ready = np.load("./first_fc_layer.npz")
second_fc_ready = np.load("./second_fc_layer.npz")

# Combine
final_fc_layers = np.concatenate((first_fc_ready['arr_0'], second_fc_ready['arr_0']))

# Save output
np.savez("./final_fc_layer.npz", final_fc_layers)