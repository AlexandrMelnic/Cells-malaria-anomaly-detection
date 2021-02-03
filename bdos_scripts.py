import tensorflow_addons as tfa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

class ImagePostProcessing:
    '''
        Class that performs the image post-processing. All the steps
        are explained in the report.
    '''

    def __init__(self,model, process=True ):
        self.process = process
        self.model = model

    def filter_img(self, diff):
        mask = tf.constant([0.,1.,0.], shape=(1,1,3))
        channels_tr = tf.constant([0.83134955, 0.6531939 , 0.6195212 ]) - 0.6
        masked_color = tf.reduce_all(diff > channels_tr, 
                                                axis=[3])
        masked_color = tf.reshape(masked_color, (diff.shape[0], 64,64,1))
        diff = tfa.image.median_filter2d(diff, filter_shape=5)
        diff = tf.where(masked_color, 0, diff)
        diff = self.threshold_img(diff, 0.1,0.45)*mask
        return diff

    def delete_few_pixels(self, img):
        if len(img[img != 0]) < 10:
            img = tf.zeros_like(img, dtype=tf.float32)
        return img

    def process_img(self, obs):
        reconstructed_img = self.model(obs)  
        if self.process == False:
            return (obs-reconstructed_img)**2
        diff = tf.maximum(-obs+reconstructed_img,0)
        diff = self.filter_img(diff)

        diff = tf.map_fn(fn=lambda x: self.delete_few_pixels(x), elems=diff)
        diff = tfa.image.mean_filter2d(diff, filter_shape=7)
        return diff**0.001 

    def threshold_img(self, diff, t1, t2):
        filtered_img = tf.where(diff < t1, 0, diff)
        filtered_img = tf.where(filtered_img > t2, 0, filtered_img)
        return filtered_img
    
    def compute_diff(self, data):
        mae = []
        for obs in data:
            diff = self.process_img(obs)
            diff = tf.reduce_mean(diff, axis=[1,2,3]).numpy()
            mae = mae + list(diff)
        return  np.array(mae)

'''
The following are just useful functions needed to make plots
and create gifs.
'''

def plot_diff(data, model):
    img = next(iter(data))
    img = tf.reshape(img, (1,64,64,3))
    post_processing = ImagePostProcessing(model)
    diff = post_processing.process_img(img)
    #diff = process_img(img)
    diff = tf.reshape(diff, (64,64,3))
    img = tf.reshape(img, (64,64,3))
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.grid(False)
    plt.title('original')
    plt.subplot(1,2,2)
    plt.imshow(tf.reshape(img, (64,64,3))+diff, alpha=1)
    plt.grid(False)
    plt.title('highlighted')

def plot_imgs(data, model):
    img = next(iter(data))
    reconstructed_img = model(tf.reshape(img, (1, 64,64,3)))
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.grid(False)
    plt.title('original')
    plt.subplot(1,2,2)
    plt.imshow(tf.reshape(reconstructed_img, (64,64,3)))
    plt.grid(False)
    plt.title('reconstructed')

def save_post_processing(data, model):
    original_img = next(iter(data))
    reconstructed_img = model(original_img)
    post_processing = ImagePostProcessing(model, process=True)
    processing_steps = []
    processing_steps.append(original_img)
    diff = tf.maximum(reconstructed_img - original_img, 0)
    processing_steps.append(diff)
    mask = tf.constant([0.,1.,0.], shape=(1,1,3))
    channels_tr = tf.constant([0.83134955, 0.6531939 , 0.6195212 ]) - 0.6
    masked_color = tf.reduce_all(diff > channels_tr, 
                                            axis=[3])
    masked_color = tf.reshape(masked_color, (diff.shape[0], 64,64,1))
    diff = tfa.image.median_filter2d(diff, filter_shape=5)
    processing_steps.append(diff)
    diff = tf.where(masked_color, 0, diff)
    processing_steps.append(diff)
    diff = post_processing.threshold_img(diff, 0.1,0.45)*mask
    processing_steps.append(diff)
    diff = tf.map_fn(fn=lambda x: post_processing.delete_few_pixels(x), elems=diff)
    processing_steps.append(diff)
    diff = tfa.image.mean_filter2d(diff, filter_shape=7)
    processing_steps.append(diff)
    diff = diff**0.001 
    processing_steps.append(diff)
    return processing_steps

def generate_gif(img_list):
    fig = plt.figure(figsize=(5,5))
    images=[]
    for i in range(len(img_list)):
        img = img_list[i]
        img_plot = plt.imshow(img[0])
        images.append([img_plot])

    my_anim = animation.ArtistAnimation(fig, images, interval=1000, blit=True, repeat_delay=500)
    plt.grid(False)
    plt.close()
    return my_anim
