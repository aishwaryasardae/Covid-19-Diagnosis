import os
import numpy as np
import pickle
#import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys

def build_model():
    try:

        EPOCHS = 10
        INIT_LR = 1e-3
        BS = 64
        default_image_size = tuple((128, 128))
        image_size = 0
        width = 128
        height = 128
        depth = 3
        print("[INFO] Loading Training dataset images...")
        DIRECTORY = "..\\covid\\dataset\\Train"
        CATEGORIES = ['Covid', 'Normal']

        data = []
        clas = []

        for category in CATEGORIES:
            print(category)
            path = os.path.join(DIRECTORY, category)
            #print(path)

            for img in os.listdir(path):
                img_path = os.path.join(path, img)
                # print(img_path)
                img = image.load_img(img_path, target_size=(128, 128))
                img = image.img_to_array(img)
                # img = img / 255
                data.append(img)
                clas.append(category)
        # print(clas)
        label_binarizer = LabelBinarizer()
        image_labels = label_binarizer.fit_transform(clas)
        pickle.dump(label_binarizer, open('label_transform.pkl_cnn', 'wb'))
        n_classes = len(label_binarizer.classes_)
        print(n_classes)
        np_image_list = np.array(data, dtype=np.float16) / 225.0

        x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state=42)

        aug = ImageDataGenerator(
            rotation_range=25, width_shift_range=0.1,
            height_shift_range=0.1, shear_range=0.2,
            zoom_range=0.2, horizontal_flip=True,
            fill_mode="nearest")
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(n_classes))
        model.add(Activation("softmax"))

        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        # distribution
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        # train the network
        print("[INFO] training network...")

        history = model.fit_generator(
            aug.flow(x_train, y_train, batch_size=BS),
            validation_data=(x_test, y_test),
            steps_per_epoch=len(x_train) // BS,
            epochs=EPOCHS, verbose=1
        )

        train_acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(train_acc) + 1)
        # Train and validation accuracy
        plt.plot(epochs, train_acc, 'b', label='Training accurarcy')
        plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
        plt.title('Training and Validation accurarcy Of CNN')

        plt.legend()
        plt.savefig('cnn_accuracy.png')

        plt.figure()
        # Train and validation loss
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and Validation loss Of CNN')
        plt.legend()
        plt.savefig('cnn_loss.png')
        plt.show()

        print("[INFO] Calculating model accuracy")
        scores = model.evaluate(x_test, y_test)
        print("Test Accuracy:", {scores[1] * 100})
        # cnn_accuracy={scores[1]*100}'''

        # save the model to disk
        print("[INFO] Saving model...")

        if os.path.exists("cnn.model.h5"):
            pass

        else:
            model.save('cnn.model.h5')

    except Exception as e:
        print("Error=", e)
        tb = sys.exc_info()[2]
        print(tb.tb_lineno)

build_model()

#Test Accuracy: {94.82758497369701}