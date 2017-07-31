# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:28:42 2017

@author: jessi
"""


import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import dataset2_g
import random
import os
import xlwt,xlrd

#rmb to restart kernel!!!!!!!!

#this is an object classification on grayscale imags and we don't need to distinguish tools of different colors.

# This code was modified from a tutorial found online, able to recognise cats from dogs.
# http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/

# Not many elements were changed, apart from:
# - the number of convolutional layers (many layers were added)
# - At first, every 128*128 images were stored in an array, one for the training pictures and one for the testing pictures.
# This does not work with large amounts of images, and with images larger than 128*128. Now, only the batch of images fed to the CNN is stored in an array.

# Convolutional Layer 1.
filter_size1 = 5
num_filters1 = 16

# Convolutional Layer 2.
filter_size2 = 5
num_filters2 = 32

# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 32
#
filter_size4 = 3
num_filters4 = 32
#
#filter_size41 = 3
#num_filters41 = 32
#
#filter_size42 = 3
#num_filters42 = 32
#
#filter_size5 = 3
#num_filters5 = 64


# Fully-connected layer.
fc_size = 128  # Number of neurons in fully-connected layer.

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 1

# image dimensions (only squares for now)
img_size = 128
# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info
folderName = "D:\Object Detection Hands\\Tools4classes\\training_data"

classes = [n for n in os.listdir(folderName)]

#classes = ['bluescissors', 'pinceVerte', 'redscissors',"screwdriverRed"]
num_classes = len(classes)

# batch size
batch_size = 64

# validation split
validation_size = 0.2

# how long to wait after validation loss stops improving before terminating training
early_stopping = None  # use None if you don't want to implement early stoping

train_path = 'Tools4classes\\training_data'
test_path = 'Tools4classes\\testing_data'

data = dataset2_g.read_train_sets(train_path, img_size, classes, validation_size=validation_size) # data
test_imagesID, test_ids = dataset2_g.read_test_set(test_path, img_size, classes)
#print("test path", test_path)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(test_imagesID)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))

'''def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape))

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    
#images = data.train.images[0:9]
print(data.train.cls[0:2])
# Get the true classes for those images.
#cls_true = data.train.cls[0:9]

# Plot the images and labels using our helper-function above.
#plot_images(images=images, cls_true=cls_true)
'''
#def plot_images(images, cls_true, cls_pred=None):
    


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,  # The previous layer.
                   num_input_channels,  # Num. channels in prev. layer.
                   filter_size,  # Width and height of each filter.
                   num_filters,  # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def new_fc_layer(input,  # The previous layer.
                 num_inputs,  # Num. inputs from prev. layer.
                 num_outputs,  # Num. outputs.
                 use_relu=True):  # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


            
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)
# print("now layer2 input")
# print(layer_conv1.get_shape())
layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

layer_conv3, weights_conv3 = \
    new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size3,
                   num_filters=num_filters3,
                   use_pooling=True)
#
layer_conv4, weights_conv4 = \
    new_conv_layer(input=layer_conv3,
                   num_input_channels=num_filters3,
                   filter_size=filter_size4,
                   num_filters=num_filters4,
                   use_pooling=True)
    
layer_flat, num_features = flatten_layer(layer_conv4)

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)

# print("now layer3 input")
# print(layer_conv2.get_shape())
#print (layer_conv2)


#layer_conv41, weights_conv41 = \
#    new_conv_layer(input=layer_conv4,
#                   num_input_channels=num_filters41,
#                   filter_size=filter_size4,
#                   num_filters=num_filters4,
#                   use_pooling=True)
#
#layer_conv42, weights_conv42 = \
#    new_conv_layer(input=layer_conv41,
#                   num_input_channels=num_filters42,
#                   filter_size=filter_size4,
#                   num_filters=num_filters4,
#                   use_pooling=True)
#
#layer_conv5, weights_conv5 = \
#    new_conv_layer(input=layer_conv42,
#                   num_input_channels=num_filters4,
#                   filter_size=filter_size5,
#                   num_filters=num_filters5,
#                   use_pooling=True)

# print("now layer flatten input")
# print(layer_conv3.get_shape())

#layer_flat, num_features = flatten_layer(layer_conv5)

#layer_fc1 = new_fc_layer(input=layer_flat,
#                         num_inputs=num_features,
#                         num_outputs=fc_size,
#                         use_relu=True)
#
#layer_fc2 = new_fc_layer(input=layer_fc1,
#                         num_inputs=fc_size,
#                         num_outputs=num_classes,
#                         use_relu=False)

y_pred = tf.nn.softmax(layer_fc2, name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)



optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
session=tf.Session()
session.run(tf.global_variables_initializer()) # for newer versions
##session.run(tf.initialize_all_variables())  # for older versions
train_batch_size = batch_size#16

num_iterations=1600
#write to excel
#sheet.write(row,col,content)
#row and column start from 0
write_to_xl=True
if write_to_xl:
    workbook=xlwt.Workbook(encoding="utf-8")
    sheet1=workbook.add_sheet("Sheet1")
    #sheet1=workbook.sheet_by_index(0)
    sheet1.write(0,0,time.strftime("%d/%m/%Y"))
    sheet1.write(0,1,time.strftime("%H:%M:%S"))
    sheet1.write(1,0,"Batch Size")
    sheet1.write(1,1,batch_size)
    
    
    sheet1.write(3,0,"Architecture")
    #filter1 size and num
    sheet1.write(3,1,"Filter 1 Size")
    sheet1.write(4,1,filter_size1)
    sheet1.write(5,1,"Filter 1 Number")
    sheet1.write(6,1,num_filters1)
    #filter1 size and num
    sheet1.write(3,2,"Filter 2 Size")
    sheet1.write(4,2,filter_size2)
    sheet1.write(5,2,"Filter 2 Number")
    sheet1.write(6,2,num_filters2)
    #number of iterations
    sheet1.write(7,0,"Number of iterations")
    sheet1.write(7,1,num_iterations)
    
    sheet1.write(8,0,"Epoch")
    sheet1.write(8,1,"Training Acc")
    sheet1.write(8,2,"Validation Acc")
    sheet1.write(8,3,"Validation Loss")
    sheet1.col(0).width=5000
    sheet1.col(1).width=5000 
    sheet1.col(2).width=5000
    sheet1.col(3).width=5000  
           
             
def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss,sheetn,write_to_xl):
    # Calculate the accuracy on the training-set.
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))
    if write_to_xl:
        sheetn.write(epoch+9,0,epoch+1)
        sheetn.write(epoch+9,1,str(round((float(acc)*100),2))+"%")
        sheetn.write(epoch+9,2,str(round((float(val_acc)*100),2))+"%")
        sheetn.write(epoch+9,3,round(float(val_loss),4))
    
    

def show_time(time_in_seconds):
    hours=int(time_in_seconds/3600)
    minutes=int((time_in_seconds-3600*hours)/60)
    seconds=int(time_in_seconds-3600*hours-60*minutes)
    
    print("Time spent so far: %02d:%02d:%02d"%(hours,minutes,seconds))
    
   
total_iterations = 0

def optimize(num_iterations,
             save_session=True):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    #best_val_loss = float("inf")

    start = time.time()
    for i in range(total_iterations,
                   total_iterations + num_iterations):
        # create log writer object



        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, flattened image shape]
        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)
        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        feed_dict_validate = {x: x_valid_batch,
                              y_true: y_valid_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)
        #saver = tf.train.Saver()
        #saver.save(session, 'D:\\Object Detection Hands\\my_test_model')

        # Print status at end of each epoch (defined as full pass through training dataset).
        #if i % int(data.train.num_examples / batch_size) == 0:
        if i % int( batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / batch_size)

            print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss,sheet1,write_to_xl)
            show_time(time.time()-start)
        
    if save_session:
    # Update the total number of iterations performed.
        total_iterations += num_iterations
        saver = tf.train.Saver()
        saver.save(session, 'D:\\Object Detection Hands\\1600with4layers_grey')
        print('Saved')
        
'''test_batch_size=256
def print_test_accuracy():

    # Number of images in the test-set.
    num_test = len(test_imagesID)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))'''


optimize(num_iterations)
# print_validation_accuracy()
if write_to_xl:
    workbook.save("pythonxl%s.xls"%(time.strftime("%d%m%Y")))
    print("Excel created")
