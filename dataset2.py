import os
import glob
import numpy as np
import cv2
from sklearn.utils import shuffle
import random


def load_train(train_path, image_size, classes):
    images = []

    imagesID = []

    labels = []
    ids = []
    cls = []

    print('Reading training images')
    for fld in classes:  # assuming data directory has a separate folder for each class, and that each folder is named after the class
        
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        
        path = os.path.join(train_path, fld, '*g')
        files = glob.glob(path)
        # random.shuffle(files)
        for fl in files:
            # image = cv2.imread(fl)
            # image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
            # images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            
            # Before:
            flbase = os.path.basename(fl)  # Basename
            ids.append(flbase)

            # Now :
            imagesID.append(os.path.abspath(fl))  # This is the full path of the image
            #

            cls.append(fld)
    # images = np.array(images) # bef
    labels = np.array(labels)

    # images, labels = shuffle(images, labels, random_state=0) # bef
    # Now :
    imagesID, labels = shuffle(imagesID, labels, random_state=0)

    ids = np.array(ids)
    cls = np.array(cls)

    return imagesID, labels, ids, cls
    # return images, labels, ids, cls


def load_test(test_path, image_size, classes):
    X_testPath = []
    X_test_id = []

    for class_name in classes:
        path = os.path.join(test_path, class_name, '*g')
        files = sorted(glob.glob(path))
        # files = random.shuffle(files)
        # X_test = []
        # X_test_id = []
        print("Reading test images")
        i = 0
        for fl in files:
            if i < 0:
                i += 1
                continue
            i = 0
            flbase = os.path.basename(fl)
            
            img = cv2.imread(fl)
            img = cv2.resize(img, (image_size, image_size), cv2.INTER_LINEAR)
            X_testPath.append(os.path.abspath(fl))
            X_test_id.append(flbase)
            
        #print (len(img[0])) 128
            

            ### because we're not creating a DataSet object for the test images, normalization happens here
    #X_test = np.array(X_test, dtype=np.uint8)
    #X_test = X_test.astype('float32')
    #X_test = X_test / 255

    X_testPath, X_test_id = shuffle(X_testPath, X_test_id, random_state=0)

    return X_testPath, X_test_id


class DataSet(object):
    def __init__(self, images, labels, ids, cls):
        self._num_examples = len(images) # number of images
        #images = images.astype(np.float32)
        #images = np.multiply(images, 1.0 / 255.0) #Put this in the batch

        # self._images = images

        self._imagesID = images

        self._labels = labels
        self._ids = ids
        self._cls = cls
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._imagesID
    #'D:\\Object Detection Hands\\ToolsSD\\training_data\\largeWireCutter\\largeWireCutter.1707.jpg'
    
    @property
    def labels(self):
        return self._labels
    #[ 0.  1.  0.  0.  0.  0.] len=6

    @property
    def ids(self):
        return self._ids
    #'blueScissors.6449.jpg'

    @property
    def cls(self):
        return self._cls
    #'greenScrewdriver' 

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # # Shuffle the data (maybe)
            #perm = np.arange(self._num_examples)
            #np.random.shuffle(perm)

            #self._imagesID = self._imagesID[perm]
            #self._labels = self._labels[perm]


            # Start next epoch

            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        # Creating the self.images
        Ximages = []
        for fl in self._imagesID[start:end]:
            img = cv2.imread(fl)
            Ximages.append(img)

        #print(type(Ximages), np.shape(Ximages[0]))
        Ximages = np.array(Ximages, dtype=np.uint32)
        Ximages = Ximages.astype('float32')
        Ximages = Ximages / 255

        return Ximages, self._labels[start:end], self._ids[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size=0):
    class DataSets(object):
        pass

    data_sets = DataSets()

    imagesID, labels, ids, cls = load_train(train_path, image_size, classes)
    imagesID, labels, ids, cls = shuffle(imagesID, labels, ids, cls)  # shuffle the data

    if isinstance(validation_size, float):
        #validation_size = int(validation_size * imagesID.shape[0])
        validation_size = int(validation_size * len(imagesID))

    validation_images = imagesID[:validation_size]
    validation_labels = labels[:validation_size]
    validation_ids = ids[:validation_size]
    validation_cls = cls[:validation_size]

    train_images = imagesID[validation_size:]
    train_labels = labels[validation_size:]
    train_ids = ids[validation_size:]
    train_cls = cls[validation_size:]

    data_sets.train = DataSet(train_images, train_labels, train_ids, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_ids, validation_cls)

    return data_sets


def read_test_set(test_path, image_size, classes):
    imagesID, ids = load_test(test_path, image_size, classes)
    
    return imagesID, ids