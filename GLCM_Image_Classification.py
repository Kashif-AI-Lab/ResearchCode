# https://youtu.be/5x-CIHRmMNY
"""
@author: Sreenivas Bhattiprolu
skimage.feature.greycomatrix(image, distances, angles, levels=None, symmetric=False, normed=False)
distances - List of pixel pair distance offsets.
angles - List of pixel pair angles in radians.
skimage.feature.greycoprops(P, prop)
prop: The property of the GLCM to compute.
{‘contrast’, ‘dissimilarity’, ‘homogeneity’, ‘energy’, ‘correlation’, ‘ASM’}
"""



import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import pandas as pd
from skimage.filters import sobel, roberts, scharr, prewitt
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
from skimage.measure import shannon_entropy
from skimage.feature import hog, canny
from sklearn.metrics import classification_report, confusion_matrix,  precision_recall_curve
from sklearn import metrics
from skimage.io import imread
from skimage.transform import resize 
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
import time 
from sklearn.metrics import roc_curve


#Capture images and labels into arrays.
#Start by creating empty lists.
train_images = []
train_labels = [] 
#for directory_path in glob.glob("cell_images/train/*"):
for directory_path in glob.glob("E:\Gdnask University of Technology\Conv_Mixer_Model_Finger_Vein_PAD\FVPAD_DATASET\SCUT_PAD_75\Train/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        print(img_path)
        img = cv2.imread(img_path, 0) #Reading color images
        img = cv2.resize(img, (300, 128)) #Resize images
        train_images.append(img)
        train_labels.append(label)
        
train_images = np.array(train_images)
train_labels = np.array(train_labels)

#Do exactly the same for test/validation images
# test
test_images = []
test_labels = []
#for directory_path in glob.glob("cell_images/test/*"): 
for directory_path in glob.glob("E:\Gdnask University of Technology\Conv_Mixer_Model_Finger_Vein_PAD\FVPAD_DATASET\SCUT_PAD_75\Valid/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (300, 128))
        test_images.append(img)
        test_labels.append(fruit_label)
        
test_images = np.array(test_images)
test_labels = np.array(test_labels)

#Encode labels from text (folder names) to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

#Split data into test and train datasets (already split but assigning to meaningful convention)
#If you only have one dataset then split here
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

# Normalize pixel values to between 0 and 1
#x_train, x_test = x_train / 255.0, x_test / 255.0

###################################################################

# Run the whole from here to return_the_dataset_script. 

# FEATURE EXTRACTOR function
# input shape is (n, x, y, c) - number of images, x, y, and channels
def feature_extractor(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):  #iterate through each file 
        #print(image)
        
        df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
        #Reset dataframe to blank after each loop.
        
        img = dataset[image, :,:]


################################################################
    #START ADDING DATA TO THE DATAFRAME
# Haar Feature Like Feature 

        # feature_types = ['type-2-y', 'type-3-x','type-4']
        # feat_t = feature_types[0]
        # coord, _ = haar_like_feature_coord(img.shape[0], img.shape[1], feat_t)
                    
        # haar_feature = draw_haar_like_feature(img, 0, 0,
        #                               img.shape[0],
        #                               img.shape[1],
        #                               coord,
        #                               max_n_features=5,
        #                               random_state=0)
        # haar_feature_img=haar_feature.reshape(-1)
        # df['haar_feature']=haar_feature_img
        
        
#######################################################################               
          #Full image
#GLCM Texture Feature Extraction 

        GLCM = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
        GLCM = greycomatrix(img, [1], [0])       
        GLCM_Energy = greycoprops(GLCM, 'energy')[0]
        df['Energy'] = GLCM_Energy
        GLCM_corr = greycoprops(GLCM, 'correlation')[0]
        df['Corr'] = GLCM_corr       
        # GLCM_diss = greycoprops(GLCM, 'dissimilarity')[0]
        # df['Diss_sim'] = GLCM_diss       
        # GLCM_hom = greycoprops(GLCM, 'homogeneity')[0]
        # df['Homogen'] = GLCM_hom       
        GLCM_contr = greycoprops(GLCM, 'contrast')[0]
        df['Contrast'] = GLCM_contr



        GLCM2 = greycomatrix(img, [3], [0])       
        GLCM_Energy2 = greycoprops(GLCM2, 'energy')[0]
        df['Energy2'] = GLCM_Energy2
        GLCM_corr2 = greycoprops(GLCM2, 'correlation')[0]
        df['Corr2'] = GLCM_corr2       
        # GLCM_diss2 = greycoprops(GLCM2, 'dissimilarity')[0]
        # df['Diss_sim2'] = GLCM_diss2       
        # GLCM_hom2 = greycoprops(GLCM2, 'homogeneity')[0]
        # df['Homogen2'] = GLCM_hom2       
        GLCM_contr2 = greycoprops(GLCM2, 'contrast')[0]
        df['Contrast2'] = GLCM_contr2

        # GLCM3 = greycomatrix(img, [5], [0])       
        # GLCM_Energy3 = greycoprops(GLCM3, 'energy')[0]
        # df['Energy3'] = GLCM_Energy3
        # GLCM_corr3 = greycoprops(GLCM3, 'correlation')[0]
        # df['Corr3'] = GLCM_corr3       
        # # GLCM_diss3 = greycoprops(GLCM3, 'dissimilarity')[0]
        # # df['Diss_sim3'] = GLCM_diss3       
        # # GLCM_hom3 = greycoprops(GLCM3, 'homogeneity')[0]
        # # df['Homogen3'] = GLCM_hom3       
        # GLCM_contr3 = greycoprops(GLCM3, 'contrast')[0]
        # df['Contrast3'] = GLCM_contr3


        

        # GLCM4 = greycomatrix(img, [0], [np.pi/4])       
        # GLCM_Energy4 = greycoprops(GLCM4, 'energy')[0]
        # df['Energy4'] = GLCM_Energy4
        # GLCM_corr4 = greycoprops(GLCM4, 'correlation')[0]
        # df['Corr4'] = GLCM_corr4       
        # GLCM_diss4 = greycoprops(GLCM4, 'dissimilarity')[0]
        # df['Diss_sim4'] = GLCM_diss4       
        # GLCM_hom4 = greycoprops(GLCM4, 'homogeneity')[0]
        # df['Homogen4'] = GLCM_hom4       
        # GLCM_contr4 = greycoprops(GLCM4, 'contrast')[0]
        # df['Contrast4'] = GLCM_contr4
        
        # GLCM5 = greycomatrix(img, [0], [np.pi/2])       
        # GLCM_Energy5 = greycoprops(GLCM5, 'energy')[0]
        # df['Energy5'] = GLCM_Energy5
        # GLCM_corr5 = greycoprops(GLCM5, 'correlation')[0]
        # df['Corr5'] = GLCM_corr5       
        # GLCM_diss5 = greycoprops(GLCM5, 'dissimilarity')[0]
        # df['Diss_sim5'] = GLCM_diss5       
        # GLCM_hom5 = greycoprops(GLCM5, 'homogeneity')[0]
        # df['Homogen5'] = GLCM_hom5       
        # GLCM_contr5 = greycoprops(GLCM5, 'contrast')[0]
        # df['Contrast5'] = GLCM_contr5
        
        # Add more filters as needed
        # entropy = shannon_entropy(img)
        # df['Entropy1'] = entropy
        
        # Gradient = hog(img)
        # df['HOG'] = Gradient 
        
###########################################################################################
#Gabor Feature Extraction 


#         num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
#         kernels = []
#         for theta in range(2):   #Define number of thetas
#             theta = theta / 4. * np.pi
#             for sigma in (1, 3):  #Sigma with 1 and 3
#                 for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
#                     for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
            
                
#                         gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
# #                print(gabor_label)
#                         ksize=3
#                         kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
#                         kernels.append(kernel)
#                 #Now filter the image and add values to a new column 
#                         fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
#                         filtered_img = fimg.reshape(-1)
#                         df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
#                         print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
#                         num += 1  #Increment for gabor column label


###################################################################################


# # settings for LBP
#         radius = 3
#         n_points = 8 * radius

#         lbp=local_binary_pattern(img,n_points,radius)
#         df['local_binary_pattern']=lbp
        
        #Append features from current image to the dataset
###################################################################################
        #Edge Detection Algorihtm 
        # Canny_edge=canny(img, sigma=3)
        # df['canny']=Canny_edge
        
        
        # Sobel_edge=sobel(img)
        # df['sobel']= Sobel_edge
        
        # Roberts_edge=roberts(img)
        # df['roberts']=Roberts_edge
        
        # Scharr_edge=scharr(img)
        # df['scharr']=Scharr_edge
        
        # Prewitt_edge=prewitt(img)
        # df['prewitt']=Prewitt_edge
        
        
        image_dataset = image_dataset.append(df)
        
    return image_dataset

####################################################################

#Extract features from training images
image_features = feature_extractor(x_train)
X_for_ML =image_features
#Reshape to a vector for Random Forest / SVM training
n_features = image_features.shape[1]
image_features = np.expand_dims(image_features, axis=0)
X_for_ML = np.reshape(image_features, (x_train.shape[0], -1))  #Reshape to #images, features

####################################################################
#features_list=list(X_for_ML)
#######################################################################333

####################################################################################

################################################################################
#Define the classifier
from sklearn.ensemble import RandomForestClassifier

#Can also use SVM but RF is faster and may be more accurate.

RF_model = RandomForestClassifier(n_estimators = 40, random_state = 42, max_features='log2', criterion='entropy')


# Fit the model on training data
RF_model.fit(X_for_ML, y_train) #For sklearn no one hot encoding

#Predict on Test data
#Extract features from test data and reshape, just like training data
test_features = feature_extractor(x_test)
test_features = np.expand_dims(test_features, axis=0)
test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))

#Predict on test
test_prediction = RF_model.predict(test_for_RF)

#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels_encoded, test_prediction))

print(classification_report(test_labels_encoded, test_prediction))
 
#Print confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels_encoded, test_prediction )

fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
sns.set(font_scale=3)

sns.heatmap(cm, annot=True,cmap='tab20', linewidths=0, ax=ax)
ax.xaxis.set_ticklabels(['Covid-19', 'Normal']); ax.yaxis.set_ticklabels(['Covid-19', 'Normal']);
plt.xlabel('Predicted Value',fontsize='medium')
plt.ylabel('Actual Value',fontsize='medium')

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy
accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)



# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(test_prediction, test_labels_encoded).ravel()

# Calculate APCER and BPCER
APCER = fp / (tp + fp)
BPCER = fn / (tn + fn)
# Calculate ACER
ACER = (APCER + BPCER) / 2

print("APCER: {:.2f}%".format(APCER * 100))
print("BPCER: {:.2f}%".format(BPCER * 100))
print("ACER: {:.2f}%".format(ACER * 100))



# Plotting DET Curve
test_probabilities = RF_model.predict_proba(test_for_RF)[:, 1]
fpr, fnr, thresholds = roc_curve(test_labels_encoded, test_probabilities)

plt.figure(figsize=(10, 10))
plt.plot(fpr, fnr, marker='o', label='DET Curve', color='blue')
# plt.xscale('log')
# plt.yscale('log')
plt.title('Detection Error Tradeoff (DET) Curve')
plt.xlabel('False Positi Rate (APCER)')
plt.ylabel('False Negative Rate (BPCER)')
plt.legend()
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.show()

######################################################
#Computational Cost of RF Classifier 

# Measure feature extraction time for training images
start_time_train = time.time()
train_features = feature_extractor(x_train)
end_time_train = time.time()
extraction_time_train = end_time_train - start_time_train

# Measure feature extraction time for test/validation images
start_time_test = time.time()
test_features = feature_extractor(x_test)
end_time_test = time.time()
extraction_time_test = end_time_test - start_time_test

# Reshape and preprocess test features
test_features = np.expand_dims(test_features, axis=0)
test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))

# Measure classification time
start_time_classification = time.time()
test_prediction = RF_model.predict(test_for_RF)
end_time_classification = time.time()
classification_time = end_time_classification - start_time_classification

# Calculate overall processing time
overall_processing_time = extraction_time_train + extraction_time_test + classification_time

# Print feature extraction times
print("Feature extraction time for training images: {:.2f} seconds".format(extraction_time_train))
print("Feature extraction time for test/validation images: {:.2f} seconds".format(extraction_time_test))

# Print classification and overall processing times
print("Classification time: {:.2f} seconds".format(classification_time))
print("Overall processing time: {:.2f} seconds".format(overall_processing_time))



#####################################################################################


import lightgbm as lgb
from sklearn.metrics import confusion_matrix, roc_curve, auc
 #Class names for LGBM start at 0 so reassigning labels from 1,2,3,4 to 0,1,2,3
d_train = lgb.Dataset(X_for_ML, label=y_train)

from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

param_grid = {
    'num_leaves': [31, 50, 100],
    'max_depth': [-1, 10, 20],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [20, 40, 100]
}

lgb_model = lgb.LGBMClassifier()
grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_for_ML, y_train)
print("Best parameters found: ", grid_search.best_params_)

# https://lightgbm.readthedocs.io/en/latest/Parameters.html
lgbm_params = {'learning_rate':0.07, 'boosting_type':'gbdt',    
              'objective':'binary',
              'metric': 'binary_logloss',
              'num_leaves':31,
              'max_depth':2,
              'num_class':1,
              'force_col_wise': 'true'}  #no.of unique values in the target class not inclusive of the end value


lgb_model = lgb.train(lgbm_params, d_train, 100) #50 iterations. Increase iterations for small learning rates



#Predict on Test data
#Extract features from test data and reshape, just like training data
test_features = feature_extractor(x_test)
test_features = np.expand_dims(test_features, axis=0)
test_for_lgb = np.reshape(test_features, (x_test.shape[0], -1))

#Predict on test
test_prediction = lgb_model.predict(test_for_lgb)

# Round the predicted probabilities to obtain the predicted labels
test_prediction_labels = np.round(test_prediction)


for i in range(0, x_test.shape[0]):
    if test_prediction[i]>=.5:
        test_prediction[i]=1
    else:
        test_prediction[i]=0

#test_prediction=np.argmax(test_prediction, axis=1)
#Inverse le transform to get original label back. 
#test_prediction = le.inverse_transform(test_prediction)

#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels_encoded, test_prediction))

print(classification_report(test_labels_encoded, test_prediction))
 
#Print confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels_encoded, test_prediction)

fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
sns.set(font_scale=1.5)
sns.heatmap(cm, annot=True, linewidths=1, ax=ax)
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')


# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(test_prediction, test_labels_encoded).ravel()

# Calculate APCER and BPCER
APCER = fp / (tp + fp)
BPCER = fn / (tn + fn)
# Calculate ACER
ACER = (APCER + BPCER) / 2

print("APCER: {:.2f}%".format(APCER * 100))
print("BPCER: {:.2f}%".format(BPCER * 100))
print("ACER: {:.2f}%".format(ACER * 100))


#Computational Cost of LightGBM Classifier 

# Train the LightGBM model and measure the training time
start_time_train = time.time()
lgb_model = lgb.train(lgbm_params, d_train, 100)
end_time_train = time.time()
training_time = end_time_train - start_time_train
print("Training time: {:.2f} seconds".format(training_time))

# Measure the feature extraction time for test data
start_time_test = time.time()
test_features = feature_extractor(x_test)
test_features = np.expand_dims(test_features, axis=0)
test_for_lgb = np.reshape(test_features, (x_test.shape[0], -1))
end_time_test = time.time()
feature_extraction_time_test = end_time_test - start_time_test
print("Feature extraction time for test data: {:.2f} seconds".format(feature_extraction_time_test))

# Measure the classification time
start_time_prediction = time.time()
test_prediction = lgb_model.predict(test_for_lgb)
end_time_prediction = time.time()
classification_time = end_time_prediction - start_time_prediction
print("Classification time: {:.2f} seconds".format(classification_time))

# Compute overall processing time
overall_processing_time = training_time + feature_extraction_time_test + classification_time
print("Overall processing time: {:.2f} seconds".format(overall_processing_time))





##################################################################################

import xgboost as xgb
Xgboost_model=xgb.XGBClassifier()

Xgboost_model.fit(X_for_ML, y_train)

#Predict on Test data
#Extract features from test data and reshape, just like training data
test_features = feature_extractor(x_test)
test_features = np.expand_dims(test_features, axis=0)
test_for_XgBoost = np.reshape(test_features, (x_test.shape[0], -1))

#Predict on test
test_prediction = Xgboost_model.predict(test_for_XgBoost)


for i in range(0, x_test.shape[0]):
    if test_prediction[i]>=.5:
        test_prediction[i]=1
    else:
        test_prediction[i]=0

#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels_encoded, test_prediction))

print(classification_report(test_labels_encoded, test_prediction))
 
#Print confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels_encoded, test_prediction)

fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
sns.set(font_scale=3)
sns.heatmap(cm, annot=True, cmap='tab20', linewidths=0, ax=ax)
ax.xaxis.set_ticklabels(['Covid-19', 'Normal']); ax.yaxis.set_ticklabels(['Covid-19', 'Normal']);
plt.xlabel('Predicted Value',fontsize='medium')
plt.ylabel('Actual Value',fontsize='medium')

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy
accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(test_prediction, test_labels_encoded).ravel()

# Calculate APCER and BPCER
APCER = fp / (tp + fp)
BPCER = fn / (tn + fn)
# Calculate ACER
ACER = (APCER + BPCER) / 2

print("APCER: {:.2f}%".format(APCER * 100))
print("BPCER: {:.2f}%".format(BPCER * 100))
print("ACER: {:.2f}%".format(ACER * 100))

#Computational Cost of Random Forest Classifier 

# Train the XGBoost model and measure the training time
start_time_train = time.time()
Xgboost_model.fit(X_for_ML, y_train)
end_time_train = time.time()
training_time = end_time_train - start_time_train
print("Training time: {:.2f} seconds".format(training_time))

# Extract features from test data and reshape, just like training data
start_time_test = time.time()
test_features = feature_extractor(x_test)
test_features = np.expand_dims(test_features, axis=0)
test_for_XgBoost = np.reshape(test_features, (x_test.shape[0], -1))
end_time_test = time.time()
feature_extraction_time_test = end_time_test - start_time_test
print("Feature extraction time for test data: {:.2f} seconds".format(feature_extraction_time_test))

# Predict on test data and measure the classification time
start_time_prediction = time.time()
test_prediction = Xgboost_model.predict(test_for_XgBoost)
end_time_prediction = time.time()
classification_time = end_time_prediction - start_time_prediction
print("Classification time: {:.2f} seconds".format(classification_time))

# Round the predicted probabilities to obtain the predicted labels
test_prediction = (test_prediction >= 0.5).astype(int)

# Compute overall processing time
overall_processing_time = training_time + feature_extraction_time_test + classification_time
print("Overall processing time: {:.2f} seconds".format(overall_processing_time))




################################################################################################
#CAT Boosting Algorihtm 
# import required libraries
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix


# initialize the classifier
catboost_model = CatBoostClassifier( learning_rate=0.1, depth=6, random_seed=42)


# train the model on the training data
catboost_model.fit(X_for_ML, y_train)


#Extract features from test data and reshape, just like training data
test_features = feature_extractor(x_test)
test_features = np.expand_dims(test_features, axis=0)
test_for_catboost_model = np.reshape(test_features, (x_test.shape[0], -1))



#Predict on Test data
#Extract features from test data and reshape, just like training data
test_features = feature_extractor(x_test)
test_features = np.expand_dims(test_features, axis=0)
test_for_CATBoosting = np.reshape(test_features, (x_test.shape[0], -1))


#Predict on test
test_prediction = catboost_model.predict(test_for_catboost_model)

for i in range(0, x_test.shape[0]):
    if test_prediction[i]>=.5:
        test_prediction[i]=1
    else:
        test_prediction[i]=0

#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels_encoded, test_prediction))

print(classification_report(test_labels_encoded, test_prediction))
 
#Print confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels_encoded, test_prediction)

fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
sns.set(font_scale=3)
sns.heatmap(cm, annot=True, cmap='tab20', linewidths=0, ax=ax)
ax.xaxis.set_ticklabels(['Covid-19', 'Normal']); ax.yaxis.set_ticklabels(['Covid-19', 'Normal']);
plt.xlabel('Predicted Value',fontsize='medium')
plt.ylabel('Actual Value',fontsize='medium')

total1=sum(sum(cm))
#####from confusion matrix calculate accuracy
accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(test_prediction, test_labels_encoded).ravel()

# Calculate APCER and BPCER
APCER = fp / (tp + fp)
BPCER = fn / (tn + fn)
# Calculate ACER
ACER = (APCER + BPCER) / 2

print("APCER: {:.2f}%".format(APCER * 100))
print("BPCER: {:.2f}%".format(BPCER * 100))
print("ACER: {:.2f}%".format(ACER * 100))



# Computational Cost of CAT Boosting Model 

# Start measuring feature training time
start_time = time.time()

# Extract features from training data and reshape
train_features = feature_extractor(x_train)
train_features = np.expand_dims(train_features, axis=0)
train_for_catboost_model = np.reshape(train_features, (x_train.shape[0], -1))

# Initialize the classifier
catboost_model = CatBoostClassifier(learning_rate=0.1, depth=6, random_seed=42)

# Train the model on the training data
catboost_model.fit(train_for_catboost_model, y_train)

# Calculate feature training time
feature_training_time = time.time() - start_time
print("Feature Training Time:", feature_training_time)

# Start measuring feature testing time
start_time = time.time()

# Extract features from test data and reshape
test_features = feature_extractor(x_test)
test_features = np.expand_dims(test_features, axis=0)
test_for_catboost_model = np.reshape(test_features, (x_test.shape[0], -1))

# Calculate feature testing time
feature_testing_time = time.time() - start_time
print("Feature Testing Time:", feature_testing_time)

# Start measuring classification time
start_time = time.time()

# Predict on test data
test_prediction = catboost_model.predict(test_for_catboost_model)

# Apply threshold to convert probabilities to binary predictions
test_prediction = (test_prediction >= 0.5).astype(int)

# Calculate classification time
classification_time = time.time() - start_time
print("Classification Time:", classification_time)

# Calculate overall processing time
overall_processing_time = feature_training_time + feature_testing_time + classification_time
print("Overall Processing Time:", overall_processing_time)


##############################################################################################
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Initialize the classifier
hgb_model = HistGradientBoostingClassifier(learning_rate=0.1, max_depth=6, random_state=42)

# Train the model on the training data
hgb_model.fit(X_for_ML, y_train)

# Extract features from test data and reshape, just like training data
test_features = feature_extractor(x_test)
test_features = np.expand_dims(test_features, axis=0)
test_for_hgb_model = np.reshape(test_features, (x_test.shape[0], -1))

# Predict on test data
test_prediction = hgb_model.predict(test_for_hgb_model)

# Print overall accuracy
print("Accuracy = ", accuracy_score(test_labels_encoded, test_prediction))

# Print classification report
print(classification_report(test_labels_encoded, test_prediction))

# Print confusion matrix
cm = confusion_matrix(test_labels_encoded, test_prediction)
fig, ax = plt.subplots(figsize=(10, 10))  # Sample figsize in inches
sns.set(font_scale=3)
sns.heatmap(cm, annot=True, cmap='tab20', linewidths=0, ax=ax)
ax.xaxis.set_ticklabels(['Covid-19', 'Normal'])
ax.yaxis.set_ticklabels(['Covid-19', 'Normal'])
plt.xlabel('Predicted Value', fontsize='medium')
plt.ylabel('Actual Value', fontsize='medium')

# Calculate accuracy from confusion matrix
total1 = sum(sum(cm))
accuracy1 = (cm[0, 0] + cm[1, 1]) / total1
print('Accuracy : ', accuracy1)

# Calculate sensitivity and specificity
sensitivity1 = cm[0, 0] / (cm[0, 0] + cm[0, 1])
print('Sensitivity : ', sensitivity1)

specificity1 = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print('Specificity : ', specificity1)

# Calculate confusion matrix elements
tn, fp, fn, tp = confusion_matrix(test_labels_encoded, test_prediction).ravel()

# Calculate APCER and BPCER
APCER = fp / (fp + tn)
BPCER = fn / (fn + tp)

# Calculate ACER
ACER = (APCER + BPCER) / 2

print("APCER: {:.2f}%".format(APCER * 100))
print("BPCER: {:.2f}%".format(BPCER * 100))
print("ACER: {:.2f}%".format(ACER * 100))
#################################################################################

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time

# Initialize the classifier
hgb_model = HistGradientBoostingClassifier(learning_rate=0.1, max_depth=6, random_state=42)

# Start measuring feature training time
start_time = time.time()

# Train the model on the training data
hgb_model.fit(X_for_ML, y_train)

# Calculate feature training time
feature_training_time = time.time() - start_time
print("Feature Training Time:", feature_training_time)

# Start measuring feature testing time
start_time = time.time()

# Extract features from test data and reshape, just like training data
test_features = feature_extractor(x_test)
test_features = np.expand_dims(test_features, axis=0)
test_for_hgb_model = np.reshape(test_features, (x_test.shape[0], -1))

# Calculate feature testing time
feature_testing_time = time.time() - start_time
print("Feature Testing Time:", feature_testing_time)

# Start measuring classification time
start_time = time.time()

# Predict on test data
test_prediction = hgb_model.predict(test_for_hgb_model)

# Calculate classification time
classification_time = time.time() - start_time
print("Classification Time:", classification_time)

# Calculate overall processing time
overall_processing_time = feature_training_time + feature_testing_time + classification_time

# Print overall accuracy
print("Accuracy = ", accuracy_score(test_labels_encoded, test_prediction))

# Print classification report
print(classification_report(test_labels_encoded, test_prediction))

# Print confusion matrix
cm = confusion_matrix(test_labels_encoded, test_prediction)
fig, ax = plt.subplots(figsize=(10, 10))  # Sample figsize in inches
sns.set(font_scale=3)
sns.heatmap(cm, annot=True, cmap='tab20', linewidths=0, ax=ax)
ax.xaxis.set_ticklabels(['Covid-19', 'Normal'])
ax.yaxis.set_ticklabels(['Covid-19', 'Normal'])
plt.xlabel('Predicted Value', fontsize='medium')
plt.ylabel('Actual Value', fontsize='medium')

# Calculate accuracy from confusion matrix
total1 = sum(sum(cm))
accuracy1 = (cm[0, 0] + cm[1, 1]) / total1
print('Accuracy : ', accuracy1)

# Calculate sensitivity and specificity
sensitivity1 = cm[0, 0] / (cm[0, 0] + cm[0, 1])
print('Sensitivity : ', sensitivity1)

specificity1 = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print('Specificity : ', specificity1)

# Calculate confusion matrix elements
tn, fp, fn, tp = confusion_matrix(test_labels_encoded, test_prediction).ravel()

# Calculate APCER and BPCER
APCER = fp / (fp + tn)
BPCER = fn / (fn + tp)

# Calculate ACER
ACER = (APCER + BPCER) / 2

print("APCER: {:.2f}%".format(APCER * 100))
print("BPCER: {:.2f}%".format(BPCER * 100))
print("ACER: {:.2f}%".format(ACER * 100))
print("Overall Processing Time:", overall_processing_time)



###############################################################################################
#### DET Curve 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

# Set figure size
plt.rcParams["figure.figsize"] = (10, 8)

# Prepare plots
fig, ax_det = plt.subplots()

# Set style

sns.set_style("darkgrid")
sns.set_style("ticks")
# Function to plot DET curve for a model
def plot_det_curve(model, test_data, true_labels, label, color):
    # Get predicted probabilities for the positive class
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(test_data)[:, 1]
    else:
        # For models like LightGBM, use predict with output_margin=True
        probabilities = model.predict(test_data)

    # Calculate FPR and FNR
    fpr, fnr, thresholds = roc_curve(true_labels, probabilities)

    # Plotting the DET curve
    ax_det.plot(fpr, fnr, label=label, color=color)

# Plot DET curves for each model
plot_det_curve(catboost_model, test_for_catboost_model, test_labels_encoded,
                label='Proposed Classifier (CatBoost)', color='black')
plot_det_curve(RF_model, test_for_RF, test_labels_encoded,
                label='Random Forest', color='green')
plot_det_curve(Xgboost_model, test_for_XgBoost, test_labels_encoded,
                label='XgBoost', color='red')
plot_det_curve(lgb_model, test_for_lgb, test_labels_encoded,
                label='LightGBM', color='orange')

# Set axis to logarithmic scale for better visualization
# ax_det.set_xscale('log')
# ax_det.set_yscale('log')

# Set titles and labels
ax_det.set_title("Detection Error Tradeoff (DET) Curves", fontsize=20)
ax_det.set_xlabel("False Positive Rate (APCER)", fontsize=20)
ax_det.set_ylabel("False Negative Rate (BPCER)", fontsize=20)

# Set limits
plt.xlim([-0.02, 1.02])
plt.ylim([0.93, 1.002])

# Add grid and legend
ax_det.grid(True, which="both", linestyle="--")
plt.legend(loc="lower right", fontsize=18)

# Show the plot
plt.show()

###########################################################################################
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay

# Set figure size
plt.rcParams["figure.figsize"] = (10, 8)

# Prepare plots
fig, ax_roc = plt.subplots()

# Set style
sns.set_style("darkgrid")

# Plot ROC curves for each model
RocCurveDisplay.from_estimator(catboost_model, test_for_catboost_model, test_labels_encoded, 
                               ax=ax_roc, name='CatBoosting', color="black", linestyle='-')
RocCurveDisplay.from_estimator(RF_model, test_for_RF, test_labels_encoded, 
                               ax=ax_roc, name='Random Forest', color="green", linestyle='--')
RocCurveDisplay.from_estimator(Xgboost_model, test_for_XgBoost, test_labels_encoded, 
                               ax=ax_roc, name='Proposed Classifier (XgBoost)', color="red", linestyle=':')
RocCurveDisplay.from_estimator(lgb_model, test_for_lgb, test_labels_encoded, 
                               ax=ax_roc, name='LightGBM', color="orange", linestyle='-.')

# Plot diagonal line for random chance
plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label='Random Chance')

# Set titles and labels
ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=16)
ax_roc.set_xlabel("False Positive Rate", fontsize=14)
ax_roc.set_ylabel("True Positive Rate", fontsize=14)

# Set limits
plt.xlim(-0.03, 1.02)
plt.ylim(0, 1.02)

# Add grid and legend
ax_roc.grid(linestyle="--")
plt.legend(loc="lower right", fontsize=12)

# Show the plot
plt.show()



#Precision_Recall_Curve 

from sklearn.metrics import PrecisionRecallDisplay
fig, ax_pr =plt.subplots()
plt.rcParams["figure.figsize"] = (7,6.5)
display = PrecisionRecallDisplay.from_estimator(SVM_model,test_for_SVM, test_labels_encoded, ax=ax_pr, name='Linear_SVM',color="blue")
display = PrecisionRecallDisplay.from_estimator(SVM_model_rbf, test_for_SVM_rbf, test_labels_encoded, ax=ax_pr, name='RBF_SVM',color="orange")
display = PrecisionRecallDisplay.from_estimator(SVM_model_poly, test_for_SVM_poly, test_labels_encoded,ax=ax_pr, name='Poly_SVM',color="black")
display = PrecisionRecallDisplay.from_estimator(RF_model,test_for_RF, test_labels_encoded, ax=ax_pr,color="green")
display = PrecisionRecallDisplay.from_estimator(Xgboost_model, test_for_XgBoost,test_labels_encoded, ax=ax_pr,name='ProposedClassifier(XgBoost)', color="red")
#display = PrecisionRecallDisplay.from_estimator(lgb_model, test_for_lgb,test_labels_encoded, ax=ax_pr)
#plt.plot([0, 1], [0, 1], color="navy", linestyle="--")

ax_pr.set_title("Precision-Recall curve")
ax_pr.grid(linestyle="--")
plt.ylim(0.8,1.005)
plt.xlim(0,1.02)
plt.legend(loc="lower right")
plt.show()

#DET Curve of different classifier in single figure

fig, ax_det =plt.subplots()

DetCurveDisplay.from_estimator(SVM_model, test_for_SVM, test_labels_encoded,ax=ax_det, name='Linear_SVM',color="blue" )
DetCurveDisplay.from_estimator(SVM_model_rbf, test_for_SVM_rbf, test_labels_encoded,ax=ax_det, name='RBF_SVM',color="orange" )
DetCurveDisplay.from_estimator(SVM_model_poly, test_for_SVM_poly, test_labels_encoded,ax=ax_det, name='Poly_SVM',color="black" )
DetCurveDisplay.from_estimator(RF_model, test_for_RF, test_labels_encoded,ax=ax_det,color="green")
DetCurveDisplay.from_estimator(Xgboost_model, test_for_XgBoost, test_labels_encoded,ax=ax_det,name='ProposedClassifier(XgBoost)', color="red")
#DetCurveDisplay.from_estimator(lgb_model, test_for_lgb, test_labels_encoded,ax=ax_det)
#plt.plot([0, 1], [0, 1], color="navy", linestyle="--")

ax_det.set_title("Detection Error Tradeoff (DET) curves")
ax_det.grid(linestyle="--")
plt.legend()
plt.show()





####################################################################################################
#Check results on a few random images
import random
n=random.randint(0, x_test.shape[0]-1) #Select the index of image to be loaded for testing
img = x_test[n]
plt.imshow(img)


####################################################################################################
#Extract features and reshape to right dimensions
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_features=feature_extractor(input_img)
input_img_features = np.expand_dims(input_img_features, axis=0)
input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))
#Predict
img_prediction = lgb_model.predict(input_img_for_RF)
img_prediction=np.argmax(img_prediction, axis=1)
img_prediction = le.inverse_transform([img_prediction])  #Reverse the label encoder to original name
print("The prediction for this image is: ", img_prediction)
print("Thxe actual label for this image is: ", test_labels[n])


######################################################################################


from skimage.filters import meijering, sato, frangi, hessian
import matplotlib.pyplot as plt
from skimage import io

img=io.imread("E:/SCUT/Datasets/SDU Dataset/450 images/FVR/High Quality/31.bmp",as_gray=True)
img1=plt.imshow(img,cmap=plt.cm.gray)

meijering_img=meijering(img)
plt.imshow(meijering_img,cmap=plt.cm.gray)

sato_img=sato(img)
plt.imshow(sato_img,cmap=plt.cm.gray)

frangi_img=frangi(img)
plt.imshow(frangi_img,cmap=plt.cm.gray)

hessian_img=hessian(img)
plt.imshow(hessian_img,cmap=plt.cm.gray)

################################################################################################
#Haar Feature Like Extraction 

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import resize 
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature

#resizing image
img = imread('E:/SCUT/COVID-19 Detection using GLCM Feature/preprocess/02a.jpg')
img = resize(img, (50, 50))

feature_types = ['type-2-y', 'type-3-y','type-4']

feat_t = feature_types[0]
coord, _ = haar_like_feature_coord(img.shape[0], img.shape[1], feat_t)

haar_feature = draw_haar_like_feature(img, 0, 0,
                                        img.shape[0],
                                        img.shape[1],
                                        coord,
                                        max_n_features=5,
                                        random_state=0)

plt.imshow(haar_feature)
plt.imsave("E:/SCUT/COVID-19 Detection using GLCM Feature/preprocess/haar_features.jpg", haar_feature, cmap="gray")
#plt.show()

###########################################################################
#HOG FEature Extraction 

#importing required libraries
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import matplotlib.pyplot as plt

#reading the image
img = imread('E:/SCUT/COVID-19 Detection using GLCM Feature/preprocess/02a.jpg')
plt.axis("off")
plt.imshow(img)
print(img.shape)

#resizing image
resized_img = resize(img, (128*4, 64*4))
plt.axis("off")
plt.imshow(resized_img)
plt.show()
print(resized_img.shape)

#creating hog features
fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, multichannel=True)
print(fd.shape)
print(hog_image.shape)
plt.axis("off")
plt.imshow(hog_image, cmap="gray")
plt.show()

# save the images
plt.imsave("outputs/hog_features.jpg", hog_image, cmap="gray")
