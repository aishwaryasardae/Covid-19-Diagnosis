# Covid-19-Detection-Using-DeepLearning
Diagnosing Covid using Chest X-Rays using Deep Learning techniques  

Given X-ray images of patients, the task is to build a machine learning model that detects whether a patient has COVID-19 (target value 1) or is Non-COVID (target value 0).
An AI and Deep Learning solution to assist in diagnosing Covid-19 infection through the analysis of chest X-Ray images.
## __DATASET DESCRIPTION__  
The dataset utilized in this study is a compilation of several sources, from which the NORMAL and COVID-19 INFECTED chest X-Ray images have been extracted.

__Positive Cases__: https://github.com/ieee8023/covid-chestxray-dataset  
__Normal Cases__: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia  

## __Model Details__
Three models were created:
1. CNN Model
2. VGG-16
3. VGG-19

## __ACCURACY GRAPHS__

1. ### __CNN ACCURACY GRAPH__ :

   ![cnn_accuracy](https://github.com/aishwaryasardae/Covid-19-Detection-Using-DeepLearning/assets/109073392/9acec626-45c8-4240-9b7f-43c7fc405a18)

2. ### __CNN LOSS GRAPH__ :
   
   ![cnn_loss](https://github.com/aishwaryasardae/Covid-19-Detection-Using-DeepLearning/assets/109073392/175e23f5-16ee-446a-b4b5-2b600ec7fe24)

3. ### __VGG16 ACCURACY GRAPH__ :

   ![vgg16_accuracy](https://github.com/aishwaryasardae/Covid-19-Detection-Using-DeepLearning/assets/109073392/f316f028-4dd4-4715-bf31-260901536d21)

4. ### __VGG16 LOSS GRAPH__ :

   ![vgg16_loss](https://github.com/aishwaryasardae/Covid-19-Detection-Using-DeepLearning/assets/109073392/842bff08-046c-444f-b7ad-32955aef6b15)

5. ### __VGG19 ACCURACY GRAPH__ :

   ![vgg19_accuracy](https://github.com/aishwaryasardae/Covid-19-Detection-Using-DeepLearning/assets/109073392/3f6f49c6-31c8-41d3-b6b8-c13e1e227bd4)

6. ### __VGG19 LOSS GRAPH__ :

   ![vgg19_loss](https://github.com/aishwaryasardae/Covid-19-Detection-Using-DeepLearning/assets/109073392/b0c2b623-7fb3-4158-9488-b4670872f85e)


Among the models tested, the VGG16 model demonstrated the highest accuracy, achieving 96% accuracy in detecting COVID-19 diagnosis.










