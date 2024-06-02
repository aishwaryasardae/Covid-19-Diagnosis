import os
import sys
import numpy as np
import operator
import pickle
from keras.models import Sequential, load_model
import cv2
from keras.preprocessing import image as image_utils
import numpy
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing import image
from keras import backend as K

def detection_img(test_image):
    try:
        K.clear_session()
        data = []
        img_path = test_image
        testing_img=cv2.imread(img_path)
        #cv2.imshow("test",testing_img)
        #cv2.waitKey(0)
        cv2.imwrite("../covid/static/detection.png", testing_img)
        model = load_model('vgg16_model.h5')
        test_image = image.load_img(img_path, target_size=(128, 128))
        test_image =image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 128
        prediction = model.predict(test_image)
        lb = pickle.load(open('label_transform.pkl_vgg16', 'rb'))
        #print(lb.inverse_transform(prediction)[0])
        prediction=lb.inverse_transform(prediction)[0]
        print("result",prediction)
        K.clear_session()

        return prediction

    except Exception as e:
        print("Error=", e)
        tb = sys.exc_info()[2]
        print("LINE NO: ", tb.tb_lineno)

#detection_img("benign(2).png")
