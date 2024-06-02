import pickle
import os
import sys
import numpy as np
import operator
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
#from tensorflow.keras.utils import img_to_array,load_img
from keras.models import Sequential, load_model
from keras.preprocessing import image
from keras.preprocessing import image as image_utils
import numpy
from keras.preprocessing import image
model_path = 'vgg16_model.h5'
model = load_model(model_path)
test_image = load_img('covid3.jpg', target_size=(128,128))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image /= 255
prediction = model.predict(test_image)
print(prediction)
lb = pickle.load(open('label_transform.pkl_vgg16', 'rb'))
print(lb)
print(lb.inverse_transform(prediction))