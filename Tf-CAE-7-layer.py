from __future__ import division, print_function, absolute_import
"""
Author: PinAxe

Project: Auto Encoder Example.
Build a 7 layers auto-encoder with TensorFlow Convolutional layers
and train it on noised MNIST set. Supposed to do a serious denoising.
Also it does save and restore of the model and visualisation.
The work is derived from source code by Manish Chablani see References.

References:
    https://towardsdatascience.com/autoencoders-introduction-and-implementation-3f40483b0a85
    https://hackernoon.com/autoencoders-deep-learning-bits-1-11731e200694
    
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
    https://towardsdatascience.com/autoencoders-introduction-and-implementation-3f40483b0a85

"""
import os.path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def show() : # Visualisation function
    # Testing =================================================================
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig  = np.empty((28 * n, 28 * n))
    canvas_noisy = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))

    for i in range(n):   # MNIST test set
        batch1, _ = mnist.test.next_batch(n)
        imgs = batch1[:].reshape((-1, 28, 28, 1))
        # Add random noise to the input images
        noisy_imgs = imgs+ noise_factor * np.random.randn(*imgs.shape)
        # Clip the images to be between 0 and 1
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)

        # Encode and decode the digit images 
        g = sess.run(logits, feed_dict={inputs_: noisy_imgs})
        # Display images
        for j in range(n): # Draw the original digits
           canvas_orig  [i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = np.squeeze(imgs[j])
           canvas_noisy [i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = np.squeeze(noisy_imgs[j])
           canvas_recon [i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = np.squeeze(g[j])
        #print(g.shape)
    
 

    plt.figure(figsize=(2,2)) 
    print("Original Images")
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()
    plt.figure(figsize=(2,2)) 
    print("Noisy Images")
    plt.imshow(canvas_noisy, origin="upper", cmap="gray")
    plt.show()
    plt.figure(figsize=(2,2)) 
    print("Reconstructed Images")
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()

    return()
    
#===============================================================================
# main function

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
tf.reset_default_graph() # Need to avoid Tensor_Name not found in CHeckPoint when loading a model


inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets')
### Encoder
conv1 = tf.layers.conv2d(inputs=inputs_, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 28x28x32
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
# Now 14x14x32
conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 14x14x32
maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
# Now 7x7x32
conv3 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 7x7x16
encoded = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
# Now 4x4x16
### Decoder
upsample1 = tf.image.resize_images(encoded, size=(7,7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 7x7x16
conv4 = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 7x7x16
upsample2 = tf.image.resize_images(conv4, size=(14,14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 14x14x16
conv5 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 14x14x32
upsample3 = tf.image.resize_images(conv5, size=(28,28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 28x28x32
conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 28x28x32
logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3,3), padding='same', activation=None)
#Now 28x28x1
# Pass logits through sigmoid to get reconstructed image
decoded = tf.nn.sigmoid(logits)

learning_rate = 0.001 #0.001
# Pass logits through sigmoid and calculate the cross-entropy loss
loss = tf.pow(targets_-logits, 2)
# Get cost and define the optimizer
cost = tf.reduce_mean(loss)
opt  = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
# you may try another optimiser
# opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#Training:
epochs = 10#100
batch_size = 20 #200
# Set's how much noise we're adding to the MNIST images
noise_factor = 0.5 #0.5
display_step =  30 #50
save_step = display_step*10 

with tf.Session() as sess:
 sess.run(tf.global_variables_initializer())   
 saver = tf.train.Saver()
 file_path=".\modelCAE1.ckpt"
 if os.path.isfile(file_path+".meta") :
    saver.restore(sess, file_path) 
    print(file_path,'-found')
 else:
    print(file_path,'-NOT found') 
 for e in range(epochs):
    for ii in range(mnist.train.num_examples//batch_size):
        batch,_ = mnist.train.next_batch(batch_size)
        # Get images from the batch
        imgs = batch[:].reshape((-1, 28, 28, 1))
        
        # Add random noise to the input images
        noisy_imgs = imgs+ noise_factor * np.random.randn(*imgs.shape)
        # Clip the images to be between 0 and 1
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)
        
        # Noisy images as inputs, original images as targets
        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: noisy_imgs,
                                                         targets_: imgs})
        if ii % display_step == 0 or ii == 1:
            print("Epoch: {} of {}...".format(e+1, epochs),
                  'batch ', ii,": of ",mnist.train.num_examples//batch_size,
                 "Training loss: {:.5f}".format(batch_cost))   
            show()    
        if ii % save_step == 0:    
            save_path = saver.save(sess=sess, save_path=file_path)
            print("Model saved in file: %s" % save_path)   

