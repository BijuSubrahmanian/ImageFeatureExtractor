################################################################################################
########## Extract imagefeatures with EfficientNet arch  ##################
################################################################################################
from tensorflow.keras.applications import EfficientNetB0,EfficientNetB1,EfficientNetB2,EfficientNetB3,EfficientNetB4,EfficientNetB5,EfficientNetB6,EfficientNetB7
#from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input
#from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.preprocessing.image import load_img
from imutils import paths
import numpy as np
import pickle
import random
import os
import numpy
import cv2

resdict={'EfficientNetB0':224,'EfficientNetB1':240,
         'EfficientNetB2':260,'EfficientNetB3':300,
         'EfficientNetB4':380,'EfficientNetB5':456,
        'EfficientNetB6':528,'EfficientNetB7':600}

#Load model
model =None
def loadmodel(architecture='EfficientNetB0'):
    if architecture=='EfficientNetB0':
        #print("[INFO] loading network...")
        global model
        model = EfficientNetB0(weights="imagenet", include_top=False)
    elif architecture=='EfficientNetB1':
        model = EfficientNetB1(weights="imagenet", include_top=False)
    elif architecture=='EfficientNetB2':
        model = EfficientNetB2(weights="imagenet", include_top=False)
    elif architecture=='EfficientNetB3':
        model = EfficientNetB3(weights="imagenet", include_top=False)
    elif architecture=='EfficientNetB4':
        model = EfficientNetB4(weights="imagenet", include_top=False)
    elif architecture=='EfficientNetB5':
        model = EfficientNetB5(weights="imagenet", include_top=False)
    elif architecture=='EfficientNetB6':
        model = EfficientNetB6(weights="imagenet", include_top=False)
    elif architecture=='EfficientNetB7':
        model = EfficientNetB7(weights="imagenet", include_top=False)
    return 'model intialized with ' +architecture

loadmodel('EfficientNetB4')

def getFeatures(img:numpy.ndarray, architecture='EfficientNetB4'):
    # load the ResNet50 network and initialize the label encoder
    
    le = None
    
    target_size=(resdict[architecture],resdict[architecture])
    print(target_size)


    #image = load_img(imagewithPath, target_size=target_size)
    #Resize with OpenCV
    image=cv2.resize(img,target_size)
    image = img_to_array(image)
    # preprocess the image by (1) expanding the dimensions and
    # (2) subtracting the mean RGB pixel intensity from the
    # ImageNet dataset
    image = np.expand_dims(image, axis=0)
    print('image.shape',image.shape)
    image = preprocess_input(image)
    print('After preprocess image.shape',image.shape)
    features = model.predict([image], batch_size=32)#config.BATCH_SIZE)
    #features = features.reshape((features.shape[0], 7 * 7 * 2048))
    #Converting into 1 dimensional array / list
    print('features.shape',features.shape)
    if architecture=='EfficientNetB0':
        featurelist = features.reshape((features.shape[0], 7 * 7 * 1280))
    elif architecture=='EfficientNetB1':
        featurelist = features.reshape((features.shape[0], 7 * 7 * 1280))
    elif architecture=='EfficientNetB2':
        featurelist = features.reshape((features.shape[0], 8 * 8 * 1408))
    elif architecture=='EfficientNetB3':
        featurelist = features.reshape((features.shape[0], 9 * 9 * 1536))
    elif architecture=='EfficientNetB4':
        featurelist = features.reshape((features.shape[0], 11 *11 * 1792))
    elif architecture=='EfficientNetB5':
        featurelist = features.reshape((features.shape[0], 14* 14 * 2048))
    elif architecture=='EfficientNetB6':
        featurelist = features.reshape((features.shape[0], 16 * 16 * 2304))
    elif architecture=='EfficientNetB7':
        featurelist = features.reshape((features.shape[0], 18 * 18 * 2560))


    print('len(featurelist)',len(featurelist[0]))

    #return list(featurelist[0])
    return featurelist[0].tolist()


