# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 06:20:41 2023

@author: ok
"""

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
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics import classification_report, confusion_matrix,  precision_recall_curve

from skimage.io import imread
from skimage.transform import resize 



#SIZE = image.resize((80,240))

# from PIL import Image
# import os

# # Set the input and output directories
# input_dir = 'E:\Gdnask University of Technology\Finger Vein Presentation Attack Detection System Using Deep Learning Model\SCUTFVDF\FAKE'
# output_dir = 'E:\Gdnask University of Technology\Finger Vein Presentation Attack Detection System Using Deep Learning Model\SCUTFVD_RESIZE\FAKE'

# # Set the desired size of the images
# desired_size = (140, 160)

# # Loop through all the files in the input directory
# for filename in os.listdir(input_dir):
#     # Open the image
#     with Image.open(os.path.join(input_dir, filename)) as im:
#         # Resize the image
#         im_resized = im.resize(desired_size)
#         # Save the resized image to the output directory
#         im_resized.save(os.path.join(output_dir, filename))


#Capture images and labels into arrays.
#Start by creating empty lists.
train_images = []
train_labels = [] 
#for directory_path in glob.glob("cell_images/train/*"):
for directory_path in glob.glob("E:\Gdnask University of Technology\Finger Vein Presentation Attack Detection System Using Deep Learning Model\SCUTFVD_RESIZE/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        print(img_path)
        img = cv2.imread(img_path, 0) #Reading color images
        #img = cv2.resize(img, (140, 160)) #Resize images
        train_images.append(img)
        train_labels.append(label)
        
train_images = np.array(train_images)
train_labels = np.array(train_labels)


#Encode labels from text (folder names) to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

#Split data into test and train datasets (already split but assigning to meaningful convention)
#If you only have one dataset then split here
#x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

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
        GLCM_hom = greycoprops(GLCM, 'homogeneity')[0]
        df['Homogen'] = GLCM_hom       
        # GLCM_contr = greycoprops(GLCM, 'contrast')[0]
        # df['Contrast'] = GLCM_contr



        GLCM2 = greycomatrix(img, [3], [0])       
        GLCM_Energy2 = greycoprops(GLCM2, 'energy')[0]
        df['Energy2'] = GLCM_Energy2
        GLCM_corr2 = greycoprops(GLCM2, 'correlation')[0]
        df['Corr2'] = GLCM_corr2       
        # GLCM_diss2 = greycoprops(GLCM2, 'dissimilarity')[0]
        # df['Diss_sim2'] = GLCM_diss2       
        GLCM_hom2 = greycoprops(GLCM2, 'homogeneity')[0]
        df['Homogen2'] = GLCM_hom2       
        # GLCM_contr2 = greycoprops(GLCM2, 'contrast')[0]
        # df['Contrast2'] = GLCM_contr2
        
        image_dataset = image_dataset.append(df)
        
    return image_dataset

###############################################################################

#Extract features from training images
image_features = feature_extractor(train_images)
X_for_ML =image_features
#Reshape to a vector for Random Forest / SVM training
n_features = image_features.shape[1]
image_features = np.expand_dims(image_features, axis=0)
X_for_ML = np.reshape(image_features, (train_images.shape[0], -1))  #Reshape to #images, features

###############################################################################

################################################################################

# perform k-fold cross-validation and return all the scores
#Define the classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold,cross_validate
#Can also use SVM but RF is faster and may be more accurate.


# initialize k-fold cross-validation
k = 5 # number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=20)

RF_model = RandomForestClassifier(n_estimators = 30, random_state = 42)

scoring = {'accuracy': 'accuracy', 
           'precision': 'precision', 
           'recall': 'recall',
           'f1_score': 'f1'}

# perform k-fold cross-validation
scores = cross_validate(RF_model, X_for_ML, train_labels_encoded, cv=kf, scoring=scoring)
# Print all the scores
for i in range(k):
    print(f"Fold {i+1}:")
    print(f"Accuracy: {scores['test_accuracy'][i]}")
    print(f"Precision: {scores['test_precision'][i]}")
    print(f"Recall: {scores['test_recall'][i]}")
    print(f"F1-score: {scores['test_f1_score'][i]}")
    print()

# Print the mean and standard deviation of the scores
print("Mean Accuracy: %0.2f (+/- %0.2f)" % (np.mean(scores['test_accuracy']), np.std(scores['test_accuracy']) * 2))
print("Mean Precision: %0.2f (+/- %0.2f)" % (np.mean(scores['test_precision']), np.std(scores['test_precision']) * 2))
print("Mean Recall: %0.2f (+/- %0.2f)" % (np.mean(scores['test_recall']), np.std(scores['test_recall']) * 2))
print("Mean F1-score: %0.2f (+/- %0.2f)" % (np.mean(scores['test_f1_score']), np.std(scores['test_f1_score']) * 2))

# Visualize the results using a line plot with text annotations for average scores
metric_names = ['accuracy', 'recall', 'precision', 'f1_score']
colors = ['r', 'g', 'b', 'm']

# Create a dictionary to store average values
avg_scores = {}


# Explicitly set the figure size
plt.figure(figsize=(10, 6))

# Define an offset for the y-coordinate of the annotations
y_offset =  0.008  # Adjust this value as needed for spacing

for i, metric in enumerate(metric_names):
    metric_scores = scores['test_{}'.format(metric)]
    plt.plot(metric_scores, marker='o', color=colors[i], label=metric)
    avg_score = np.mean(metric_scores)
    plt.plot([avg_score]*k, linestyle='--', color=colors[i])
    
    # Store the average score for each metric
    avg_scores[metric] = avg_score

    # Annotate the average score on the plot with an offset
    plt.text(k-1, avg_score + (i * y_offset), f'{metric}: {avg_score:.4f}', 
             color=colors[i], verticalalignment='center', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.6))

plt.legend()
plt.title('Random Forest Classifier')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(np.arange(k), np.arange(1, k+1))
plt.plot([0, k-1], [0.5, 0.5], '--k', linewidth=2)  # add a horizontal line at 0.5
plt.xlim([-0.1, k-0.9])
plt.ylim([0.80, 1])
plt.show()

################################################################
 #Grid search algorihtm to find the best hyperparameter for model.
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import KFold, GridSearchCV, cross_validate
# import numpy as np

# # create a random forest classifier
# rf = RandomForestClassifier(random_state=42)

# # define the parameter grid to search over
# param_grid = {
#     'n_estimators': [10, 30, 50],
#     'max_depth': [None, 5, 10],
#     'min_samples_split': [2, 5, 10]
# }

# # create a grid search object
# grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')

# # fit the grid search object to the data
# grid_search.fit( X_for_ML, train_labels_encoded)

# # print the best hyperparameters found
# print(grid_search.best_params_)



############### K-Cross Validation ###############################################

import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_validate

# initialize k-fold cross-validation
k = 5 # number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# initialize your LightGBM model
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 20,
    'learning_rate': 0.13,
    'feature_fraction': 0.9
}
model = lgb.LGBMClassifier(**params)

# perform k-fold cross-validation and return all the scores
scoring = {'accuracy': 'accuracy', 
           'precision': 'precision', 
           'recall': 'recall',
           'f1_score': 'f1'}
scores = cross_validate(model, X_for_ML, train_labels_encoded, cv=kf, scoring=scoring)

# Print all the scores
for i in range(k):
    print(f"Fold {i+1}:")
    print(f"Accuracy: {scores['test_accuracy'][i]}")
    print(f"Precision: {scores['test_precision'][i]}")
    print(f"Recall: {scores['test_recall'][i]}")
    print(f"F1-score: {scores['test_f1_score'][i]}")
    print()

# Print the mean and standard deviation of the scores
print("Mean Accuracy: %0.2f (+/- %0.2f)" % (np.mean(scores['test_accuracy']), np.std(scores['test_accuracy']) * 2))
print("Mean Precision: %0.2f (+/- %0.2f)" % (np.mean(scores['test_precision']), np.std(scores['test_precision']) * 2))
print("Mean Recall: %0.2f (+/- %0.2f)" % (np.mean(scores['test_recall']), np.std(scores['test_recall']) * 2))
print("Mean F1-score: %0.2f (+/- %0.2f)" % (np.mean(scores['test_f1_score']), np.std(scores['test_f1_score']) * 2))

# Visualize the results using a line plot with text annotations for average scores
metric_names = ['accuracy', 'recall', 'precision', 'f1_score']
colors = ['r', 'g', 'b', 'm']

# Create a dictionary to store average values
avg_scores = {}

plt.figure(figsize=(10, 6))
# Define an offset for the y-coordinate of the annotations
#y_offset = 0.016  # Adjust this value as needed for spacing for IDIAP
y_offset = 0.03  # Adjust this value as needed for spacing for SCUT

for i, metric in enumerate(metric_names):
    metric_scores = scores['test_{}'.format(metric)]
    plt.plot(metric_scores, marker='o', color=colors[i], label=metric)
    avg_score = np.mean(metric_scores)
    plt.plot([avg_score]*k, linestyle='--', color=colors[i])
    
    # Store the average score for each metric
    avg_scores[metric] = avg_score

    # Annotate the average score on the plot with an offset
    plt.text(k-1, avg_score + (i * y_offset), f'{metric}: {avg_score:.4f}', 
             color=colors[i], verticalalignment='center', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.6))
plt.legend()
plt.title('LightGBM Classifier')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(np.arange(k), np.arange(1, k+1))
plt.plot([0, k-1], [0.5, 0.5], '--k', linewidth=2)  # add a horizontal line at 0.5
plt.xlim([-0.1, k-0.9])
plt.ylim([0.80, 1])
plt.show()



##################################### K-Cross Validation ##############################################

import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import KFold, cross_validate

# initialize k-fold cross-validation
k = 5 # number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Define XGBoost parameters
params = {'objective': 'binary:logistic',
          'eval_metric': 'logloss',
          'eta': 0.05,
          'max_depth': 3,
          'min_child_weight': 1,
          'subsample': 0.8,
          'colsample_bytree': 0.8,
          'nthread': 4,
          'seed': 42}

Xgboost_model = xgb.XGBClassifier(**params)

# perform k-fold cross-validation and return all the scores
scoring = {'accuracy': 'accuracy', 
           'precision': 'precision', 
           'recall': 'recall',
           'f1_score': 'f1'}
scores = cross_validate(Xgboost_model, X_for_ML, train_labels_encoded, cv=kf, scoring=scoring)

# Print all the scores
for i in range(k):
    print(f"Fold {i+1}:")
    print(f"Accuracy: {scores['test_accuracy'][i]}")
    print(f"Precision: {scores['test_precision'][i]}")
    print(f"Recall: {scores['test_recall'][i]}")
    print(f"F1-score: {scores['test_f1_score'][i]}")
    print()

# Print the mean and standard deviation of the scores
print("Mean Accuracy: %0.2f (+/- %0.2f)" % (np.mean(scores['test_accuracy']), np.std(scores['test_accuracy']) * 2))
print("Mean Precision: %0.2f (+/- %0.2f)" % (np.mean(scores['test_precision']), np.std(scores['test_precision']) * 2))
print("Mean Recall: %0.2f (+/- %0.2f)" % (np.mean(scores['test_recall']), np.std(scores['test_recall']) * 2))
print("Mean F1-score: %0.2f (+/- %0.2f)" % (np.mean(scores['test_f1_score']), np.std(scores['test_f1_score']) * 2))

# Visualize the results using a line plot with text annotations for average scores
metric_names = ['accuracy', 'recall', 'precision', 'f1_score']
colors = ['r', 'g', 'b', 'm']

# Create a dictionary to store average values
avg_scores = {}

plt.figure(figsize=(10, 6))
# Define an offset for the y-coordinate of the annotations
#y_offset = 0.016   # Adjust this value as needed for spacing
y_offset = 0.018  # Adjust this value as needed for spacing for SCUT

for i, metric in enumerate(metric_names):
    metric_scores = scores['test_{}'.format(metric)]
    plt.plot(metric_scores, marker='o', color=colors[i], label=metric)
    avg_score = np.mean(metric_scores)
    plt.plot([avg_score]*k, linestyle='--', color=colors[i])
    
    # Store the average score for each metric
    avg_scores[metric] = avg_score

    # Annotate the average score on the plot with an offset
    plt.text(k-1, avg_score + (i * y_offset), f'{metric}: {avg_score:.4f}', 
             color=colors[i], verticalalignment='center', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.6))

plt.legend()
plt.title('XGBoost Classifier')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(np.arange(k), np.arange(1, k+1))
plt.plot([0, k-1], [0.5, 0.5], '--k', linewidth=2)  # add a horizontal line at 0.5
plt.xlim([-0.1, k-0.9])
plt.ylim([0.80, 1])
plt.show()



############################################# CAT K-Cross Validation ###########################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate
from catboost import CatBoostClassifier

# Define CatBoost parameters
params = {'loss_function': 'Logloss',
          'iterations': 100,
          'learning_rate': 0.13,
          'depth': 5,
          'l2_leaf_reg': 7,
          'verbose': False}

# initialize k-fold cross-validation
k = 5 # number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# initialize the CAT Boosting classifier
CAT_clf = CatBoostClassifier(**params)

# perform k-fold cross-validation and return all the scores
scoring = {'accuracy': 'accuracy', 
           'precision': 'precision', 
           'recall': 'recall',
           'f1_score': 'f1'}

# perform k-fold cross-validation
scores = cross_validate(CAT_clf, X_for_ML, train_labels_encoded, cv=kf, scoring=scoring)

# Print all the scores
for i in range(k):
    print(f"Fold {i+1}:")
    print(f"Accuracy: {scores['test_accuracy'][i]}")
    print(f"Precision: {scores['test_precision'][i]}")
    print(f"Recall: {scores['test_recall'][i]}")
    print(f"F1-score: {scores['test_f1_score'][i]}")
    print()

# Print the mean and standard deviation of the scores
print("Mean Accuracy: %0.2f (+/- %0.2f)" % (np.mean(scores['test_accuracy']), np.std(scores['test_accuracy']) * 2))
print("Mean Precision: %0.2f (+/- %0.2f)" % (np.mean(scores['test_precision']), np.std(scores['test_precision']) * 2))
print("Mean Recall: %0.2f (+/- %0.2f)" % (np.mean(scores['test_recall']), np.std(scores['test_recall']) * 2))
print("Mean F1-score: %0.2f (+/- %0.2f)" % (np.mean(scores['test_f1_score']), np.std(scores['test_f1_score']) * 2))

# Visualize the results using a line plot with text annotations for average scores
metric_names = ['accuracy', 'recall', 'precision', 'f1_score']
colors = ['r', 'g', 'b', 'm']

# Create a dictionary to store average values
avg_scores = {}

plt.figure(figsize=(10, 6))
# Define an offset for the y-coordinate of the annotations
#y_offset = 0.019  # Adjust this value as needed for spacing
y_offset = 0.02  # Adjust this value as needed for spacing for SCUT
for i, metric in enumerate(metric_names):
    metric_scores = scores['test_{}'.format(metric)]
    plt.plot(metric_scores, marker='o', color=colors[i], label=metric)
    avg_score = np.mean(metric_scores)
    plt.plot([avg_score]*k, linestyle='--', color=colors[i])
    
    # Store the average score for each metric
    avg_scores[metric] = avg_score

    # Annotate the average score on the plot with an offset
    plt.text(k-1, avg_score + (i * y_offset), f'{metric}: {avg_score:.4f}', 
             color=colors[i], verticalalignment='center', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.6))
    
plt.legend()
plt.title('CatBoost Classifier')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(np.arange(k), np.arange(1, k+1))
plt.plot([0, k-1], [0.5, 0.5], '--k', linewidth=2)  # add a horizontal line at 0.5
plt.xlim([-0.1, k-0.9])
plt.ylim([0.80, 1])
plt.show()



###############################################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

# Define HistGradientBoosting parameters
params = {
    'learning_rate': 0.1,
    'max_iter': 100,
    'max_depth': 5,
    'l2_regularization': 1.0,
    'random_state': 42
}

# initialize k-fold cross-validation
k = 5 # number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# initialize the HistGradientBoosting classifier
HGB_clf = HistGradientBoostingClassifier(**params)

# perform k-fold cross-validation and return all the scores
scoring = {'accuracy': 'accuracy', 
           'precision': 'precision_macro', 
           'recall': 'recall_macro',
           'f1_score': 'f1_macro'}

# perform k-fold cross-validation
scores = cross_validate(HGB_clf, X_for_ML, train_labels_encoded, cv=kf, scoring=scoring)

# Print all the scores
for i in range(k):
    print(f"Fold {i+1}:")
    print(f"Accuracy: {scores['test_accuracy'][i]}")
    print(f"Precision: {scores['test_precision'][i]}")
    print(f"Recall: {scores['test_recall'][i]}")
    print(f"F1-score: {scores['test_f1_score'][i]}")
    print()

# Print the mean and standard deviation of the scores
print("Mean Accuracy: %0.2f (+/- %0.2f)" % (np.mean(scores['test_accuracy']), np.std(scores['test_accuracy']) * 2))
print("Mean Precision: %0.2f (+/- %0.2f)" % (np.mean(scores['test_precision']), np.std(scores['test_precision']) * 2))
print("Mean Recall: %0.2f (+/- %0.2f)" % (np.mean(scores['test_recall']), np.std(scores['test_recall']) * 2))
print("Mean F1-score: %0.2f (+/- %0.2f)" % (np.mean(scores['test_f1_score']), np.std(scores['test_f1_score']) * 2))

# Visualize the results using a line plot with text annotations for average scores
metric_names = ['accuracy', 'recall', 'precision', 'f1_score']
colors = ['r', 'g', 'b', 'm']

# Create a dictionary to store average values
avg_scores = {}

plt.figure(figsize=(10, 6))
# Define an offset for the y-coordinate of the annotations
y_offset = 0.014  # Adjust this value as needed for spacing

for i, metric in enumerate(metric_names):
    metric_scores = scores['test_{}'.format(metric)]
    plt.plot(metric_scores, marker='o', color=colors[i], label=metric)
    avg_score = np.mean(metric_scores)
    plt.plot([avg_score]*k, linestyle='--', color=colors[i])
    
    # Store the average score for each metric
    avg_scores[metric] = avg_score

    # Annotate the average score on the plot with an offset
    plt.text(k-1, avg_score + (i * y_offset), f'{metric}: {avg_score:.4f}', 
             color=colors[i], verticalalignment='center', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.6))
plt.legend()
plt.title('HistGradientBoosting Classifier')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(np.arange(k), np.arange(1, k+1))
plt.plot([0, k-1], [0.5, 0.5], '--k', linewidth=2)  # add a horizontal line at 0.5
plt.xlim([-0.1, k-0.9])
plt.ylim([0.80, 1])
plt.show()


######################################################################################


###########################################################################################
#ROCAUC curve of differenct machine learning classifier in single figure  

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Assuming X_for_ML and train_labels_encoded are your features and labels
X_train, X_test, y_train, y_test = train_test_split(X_for_ML, train_labels_encoded, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(verbose=False, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Histogram Gradient Boosting': HistGradientBoostingClassifier(random_state=42)
}

# Train classifiers and compute ROC curve and AUC for each classifier
plt.figure(figsize=(10, 8))
colors = ['blue', 'green', 'red', 'cyan', 'magenta']

for (name, clf), color in zip(classifiers.items(), colors):
    clf.fit(X_train, y_train)
    
    if hasattr(clf, 'predict_proba'):
        y_score = clf.predict_proba(X_test)[:, 1]
    else:  # Use decision_function for SVM
        y_score = clf.decision_function(X_test)
    
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Print AUC scores
print("AUC Scores:")
for name, clf in classifiers.items():
    if hasattr(clf, 'predict_proba'):
        y_score = clf.predict_proba(X_test)[:, 1]
    else:
        y_score = clf.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    print(f"{name}: {roc_auc:.4f}")





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
