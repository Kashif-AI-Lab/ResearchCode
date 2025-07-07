# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 04:34:26 2021

@author: ok
"""

from imutils import paths
import matplotlib.pyplot as plt
import argparse
import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.applications import VGG16, VGG19, InceptionV3, Xception,DenseNet121, MobileNet  

from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D,Dropout,Flatten,Dense, Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2 
from tensorflow.keras.optimizers import Adam,SGD, Nadam,Ftrl, RMSprop, Adadelta, Adagrad, Adamax
from tensorflow.keras.utils import to_categorical,plot_model 
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc


INIT_LR = 1e-3
EPOCHS = 20
BS = 32
#dataset = "/content/Data"
dataset = "E:\Shandong University\Dr. Imran Quershi\Dataset_Xception"
#dataset = "/content/drive/My Drive/IDIAPCROPPED"
#dataset = "E:\SCUT\Covid-19\Covid-19_Data\dataset"

args={}
args["dataset"]=dataset

#################################################################################
import numpy as np
import cv2
iPaths = list(paths.list_images(args["dataset"]))  #image paths
data = []
labels = []
for iPath in iPaths:
	label = iPath.split(os.path.sep)[-2]
	image = cv2.imread(iPath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (224, 224))
	data.append(image)
	labels.append(label)
data = np.array(data) / 255.0
labels = np.array(labels)

##################################################################################
import os
Data_Dir = "E:\Shandong University\Dr. Imran Quershi\Dataset_Xception//"
#Data_Dir = "IDIAPCROPPED//"
#Data_Dir = "E:\SCUT\Covid-19\Covid-19_Data\dataset//"
os.getcwd()

Rimages = os.listdir(Data_Dir+"Normal")
Fimages = os.listdir(Data_Dir+"Disease")

###################################################################################
LB = LabelBinarizer()  #Initialize label binarizer
labels = LB.fit_transform(labels)
labels = to_categorical(labels);# print(labels)
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels,
	test_size=0.25, stratify=labels, random_state=0); print(Y_test)
trainAug = ImageDataGenerator(
  rotation_range=15,
  width_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  fill_mode="nearest")

#####################################################################################
#Pre-trained Xception Model using using First Flow with SVM

bModel = Xception(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))  #base_Model
#hModel = bModel.output #head_Model
#bModel.summary() 

# new_model=Model(bModel.input, outputs=bModel.get_layer('conv2d_18').output)
# new_model.summary()


#OLD Way to Remove Layer#
Last_layer=bModel.get_layer('block4_sepconv2')


print(Last_layer.output_shape)
Last_output=Last_layer.output
hModel = AveragePooling2D(pool_size=(4, 4))(Last_output)
#hModel = MaxPooling2D(pool_size=(2, 2))(hModel)
hModel = Flatten(name="flatten")(hModel)
hModel = Dense(64, activation="relu")(hModel)
hModel = Dense(2, kernel_regularizer='l2', activation ='linear')(hModel)

CNN_SVM_model = Model(inputs=bModel.input, outputs=hModel)
for layer in bModel.layers:
	layer.trainable = False
#plot_model(model,show_shapes=True)
CNN_SVM_model.summary()


opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
CNN_SVM_model.compile(loss="hinge", optimizer=opt,
	metrics=["accuracy"])
print("Compiling Starts")
R = CNN_SVM_model.fit_generator(
	trainAug.flow(X_train, Y_train, batch_size=BS),
	steps_per_epoch=len(X_train) // BS,
	validation_data=(X_test, Y_test),
	validation_steps=len(X_test) // BS,
	epochs=EPOCHS)


################################################################################
# plot the loss
from matplotlib import ticker
#plt.style.use('seaborn-talk')
#plt.style.use('seaborn')
plt.xlabel('Epochs')
plt.ylabel('Accuracy and Loss')
plt.plot(R.history['loss'], label='Train loss')
plt.plot(R.history['val_loss'], label='Validation loss')
plt.plot(R.history['accuracy'], label='Train acc')
plt.plot(R.history['val_accuracy'], label='Validation acc')
plt.legend(loc="right")
plt.ylim(0,1.02)
plt.xlim(0,10)
plt.show()

#plt.savefig('LossVal_loss')
################################################################################
#Train Accuracy and Test Accuracy/ Validation Accuracy and Test Accuracy
from matplotlib import ticker
plt.xlabel('Epochs')
plt.ylabel('Accuracy and Loss')
plt.plot(R.history['loss'], label='Train loss')
plt.plot(R.history['accuracy'], label='Train acc')
plt.legend(loc="right")
plt.ylim(0,1.02)
plt.xlim(0,10)
plt.show()


from matplotlib import ticker
plt.xlabel('Epochs')
plt.ylabel('Accuracy and Loss')
plt.plot(R.history['val_loss'], label='Validation loss')
plt.plot(R.history['val_accuracy'], label='Validation acc')
plt.legend(loc="right")
plt.ylim(0,1.02)
plt.xlim(0,10)
plt.show()


################################################################################
#Prediction Result of Xception model using First flow with SVM 

CNN_SVM_pred = CNN_SVM_model.predict(X_test, batch_size=BS)
#print(y_pred)
#for i in range(len(X_test)):
#	print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
CNN_SVM_pred = np.argmax(CNN_SVM_pred, axis=1)
print(classification_report(Y_test.argmax(axis=1), CNN_SVM_pred,
	target_names=LB.classes_))
##############################################################################

import seaborn as sns
cm = confusion_matrix(Y_test.argmax(axis=1), CNN_SVM_pred)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print(cm)
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

###############################################################################



#Input shape to the model is 224 x 224. SO resize input image to this shape.
from keras.preprocessing.image import load_img, img_to_array
img = load_img('E:\SCUT\Deep Learning Based  Presentation Attack Detection for Finger Vein Recognition\IDIAPFULL\REAL/001_L_1.png', target_size=(150, 450)) #VGG user 224 as input

# convert the image to an array
img = img_to_array(img)
# expand dimensions to match the shape of model input
img = np.expand_dims(img, axis=0)



#####################################################################################
# For Applying Machine LEarning with CNN model 
#Now, let us apply feature extractor to our training data
# features = new_model.predict(X_test)
features = new_model.predict(img)

#####################################################################################

#Plot features to view them
plt.figure(figsize=(20,20))
square = 4
ix=1
for _ in range(square):
    for _ in range(square):
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(features[0,:,:,ix-1], cmap='gray')
        ix +=1
plt.show()

######################################################################################

#Reassign 'features' as X to make it easy to follow
X=features
X = X.reshape(-1, X.shape[3])  #Make it compatible for Random Forest and match Y labels
print(X.shape)

#####################################################################################

#Reshape Y to match X
Y =Y_train.reshape(-1)
print(Y.shape)

#######################################################################################

np.unique(Y)
#########################


import xgboost as xgb
Xgboost_model=xgb.XGBClassifier()

#######################################################################################

Xgboost_model.fit(X, Y)
