# Model 2: Standard CNN

#Check GPU
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

#GPU count and name
!nvidia-smi -L

#mount google drive where dataset is saved
from google.colab import drive
drive.mount("/content/gdrive")

#Copy gdrive file to local folder. This reduces latency a lot!
!mkdir train_local
!cp /content/gdrive/MyDrive/GTSRB/gtsrb.zip /content/train_local

#unzip dataset
!unzip -u "/content/train_local/gtsrb.zip" -d "/content/train_local"

#Import libraries and dependencies
!pip install tf-explain
!pip install adversarial-robustness-toolbox

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, concatenate, BatchNormalization, MaxPool2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from tensorflow.keras.metrics import CategoricalAccuracy

#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import time
import pickle

from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import CarliniL2Method
from art.attacks.evasion import SaliencyMapMethod
from art.attacks.evasion import DeepFool
from art.estimators.classification import TensorFlowV2Classifier
from tensorflow.keras.losses import CategoricalCrossentropy

from tf_explain.core.grad_cam import GradCAM

#Specify the path to training folder and generating the label of training data
train_dir = '/content/train_local/Train/' # Change the path relative to your computer
classes = [str(x) for x in np.linspace(0,43,43).astype(int)]

#Generate training and validation data with image generator
def image_generator(train_parent_directory):
    
    train_datagen = ImageDataGenerator(rescale=1/255, validation_split = 0.2)
    
    train_generator = train_datagen.flow_from_directory(train_parent_directory,
                                  target_size = (30,30),
                                  batch_size = 100,
                                  class_mode = 'categorical',
                                  classes = classes,
                                  subset='training')
 
 
    val_generator = train_datagen.flow_from_directory(train_parent_directory,
                                 target_size=(30,30),
                                 batch_size = 25,
                                 class_mode = 'categorical',
                                 classes = classes,
                                 subset='validation')    
  
    
    return train_generator, val_generator

train_generator, val_generator = image_generator(train_dir)

x_train, y_train = next(train_generator)
x_val, y_val  = next(val_generator)

example_images.shape

"""## Visualize few examples of training data"""

f, axs = plt.subplots(1, 8, figsize=(15, 4))
for j in range(len(axs)):
    axs[j].imshow(x_train[j], cmap='binary')
    axs[j].axis('off')

f, axs = plt.subplots(1, 8, figsize=(15, 4))
for j in range(len(axs)):
    axs[j].imshow(x_train[j+8], cmap='binary')
    axs[j].axis('off')

#Generate Test Data

y_test=pd.read_csv("/content/train_local/Test.csv") #replace with data path in your computer
labels=y_test['Path'].to_numpy()
y_test=y_test['ClassId'].values

data=[]


for f in labels:
    path = '/content/train_local/Test/'+f.replace('Test/', '') #replace with data path in your computer
    img = image.load_img(path, target_size=(30, 30))
    img = image.img_to_array(img)
    data.append(np.array(img))

X_test=np.array(data)
X_test = X_test.astype('float32')/255

"""# Model 2: Standard CNN Model"""

def build_cnn():
    """
    Build CNN. The last layer must be logits instead of softmax.
    Return a compiled Keras model.
    """

    l2_reg = L2()

    # Build model
    inpt = Input(shape=(30,30,3))
    conv1 = Conv2D(
        32, (5, 5), padding='same', activation='relu')(inpt)
    drop1 = Dropout(rate=0.1)(conv1)
    conv2 = Conv2D(
        32, (5, 5), padding='same', activation='relu')(drop1)
    drop2 = Dropout(rate=0.2)(conv2)
    pool1 = MaxPool2D(pool_size=(2, 2))(drop2)

    conv3 = Conv2D(
        64, (5, 5), padding='same', activation='relu')(pool1)
    drop3 = Dropout(rate=0.3)(conv3)
    conv4 = Conv2D(
        64, (5, 5), padding='same', activation='relu')(drop3)
    drop4 = Dropout(rate=0.3)(conv4)
    pool2 = MaxPool2D(pool_size=(2, 2))(drop4)

    flat = Flatten()(pool2)
    dense1 = Dense(200, activation='relu',
                                kernel_regularizer=l2_reg)(flat)
    drop5 = Dropout(rate=0.5)(dense1)
    dense2 = Dense(200, activation='relu',
                                kernel_regularizer=l2_reg)(drop5)
    drop6 = Dropout(rate=0.5)(dense2)
    output = Dense(
        43, activation=None, kernel_regularizer=l2_reg)(drop6)
    model = Model(inputs=inpt, outputs=output)

    # Specify optimizer
    adam = Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss=output_fn, metrics=['accuracy'])

    return model

def output_fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                   logits=predicted)

model_cnn = build_cnn()
model_cnn.summary()

#training

history_cnn = model_cnn.fit(
      train_generator,
      validation_data = val_generator,  
      epochs=20,
      verbose=1)

# #Save model
# model_cnn.save("/content/gdrive/MyDrive/GTSRB/models/model2_6jun7pm")
# model_cnn.save("/content/gdrive/MyDrive/GTSRB/models/model2_6jun7pm.h5")

# #Load model
from tensorflow.keras.models  import load_model
model_cnn = load_model('/content/gdrive/MyDrive/GTSRB/models/model2_6jun7pm95.34.h5',compile=False)

"""# Predict model accuracy on clean test images"""

# Predict model accuracy on clean test images
pred = np.argmax(model_cnn.predict(X_test), axis=-1)
accuracy_score(y_test, pred)
#*0.9534441805225653

"""# Adversarial Attack"""

#Specify Classifier with ART Library
cnn_classifier = TensorFlowV2Classifier(model=model_cnn, nb_classes=43, input_shape=(30,30,3), loss_object=CategoricalCrossentropy())

# Take (5) clean images from the test set to be perturbed with different adversarial attacks
x_test_adversarial = X_test#[0:5]
y_test_adversarial = y_test#[0:5]
#print(x_test_adv.shape)

#Get Model Prediction on Adversarial Images
prediction = model_cnn.predict(x_test_adversarial)

probability_pred_clean = np.max(prediction, axis=1)
class_pred_clean = np.argmax(prediction, axis=1)

#Visualize Clean and Adversarial Images and Distance
def calculateDistance(img1, img2):
    return np.sum((img1-img2)**2)

#Initialize grad-cam for analysis later
gradcam_explainer = GradCAM()

"""## Carlini-Wagner"""

carlini_attack = CarliniL2Method(classifier = cnn_classifier, targeted=False)

start = time.time()
x_test_carlini = carlini_attack.generate(x = x_test_adversarial)
end = time.time()

print(f"1. Runtime for generating cw images with Carlini Wagner is {end - start} second")

#Save generated images into pickle file
with open('/content/gdrive/MyDrive/GTSRB/pickles/x_test_cw1000_model2.pk', 'wb') as handle:
    pickle.dump(x_test_carlini, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load pickled CW images
with open(r"/content/gdrive/MyDrive/GTSRB/pickles/for model 2/x_test_cw_model2.pk", "rb") as input_file:
  x_test_carlini = pickle.load(input_file)

# Predict model accuracy on CW images
pred_cw = np.argmax(model_cnn.predict(x_test_carlini), axis=-1)
accuracy_score(y_test, pred_cw)
#0.8405384006334126

# Predict model accuracy on CW images from model1
# Load pickled CW images
with open(r"/content/gdrive/MyDrive/GTSRB/pickles/for model 1/x_test_cw_model1.pk", "rb") as input_file:
  x_test_carlini_model1 = pickle.load(input_file)

print(x_test_carlini_model1.shape)

# Predict model accuracy
pred_cw_model1 = np.argmax(model_cnn.predict(x_test_carlini_model1), axis=-1)
accuracy_score(y_test, pred_cw_model1)

# Predict model accuracy on CW images from model3
# Load pickled CW images
with open(r"/content/gdrive/MyDrive/GTSRB/pickles/for model 3/x_test_cw_model3.pk", "rb") as input_file:
  x_test_carlini_model3 = pickle.load(input_file)

# Predict model accuracy
pred_cw_model3 = np.argmax(model_cnn.predict(x_test_carlini_model3), axis=-1)
accuracy_score(y_test, pred_cw_model3)

#Calculate average L2 distance between clean vs CW images
sum = 0
for i in range(len(x_test_adversarial)):
  sum += calculateDistance(x_test_adversarial[i], x_test_carlini[i])

avgL2_cw = sum / len(x_test_adversarial)
print("Avg L2 distance of generated CW images: ", avgL2_cw)
#0.7022802623243257
#19.729037967276536

#Predict Standard CNN Accuracy on Adversarial Images
y_pred_cw = model_cnn(x_test_carlini)
y_pred_cw = np.argmax(y_pred_cw, axis=1)
acc = sum(1 for x,y in zip(y_pred_cw, y_test_adversarial) if x == y) / float(len(y_test_adversarial))
acc

#Get Model Prediction on Adversarial Images
prediction_cw = model_cnn.predict(x_test_carlini)

probability_pred_cw = np.max(prediction_cw, axis=1)
class_pred_cw = np.argmax(prediction_cw, axis=1)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

#Visualize successful attacks

f, axs = plt.subplots(2, 5, figsize=(15, 8))
title = ['Clean Images', 'Carlini-Wagner Images']

for row, ax in enumerate(axs, start=0):
    ax[0].set_title("%s \n" % title[row], loc='left', fontsize=16)

a=0
for i,v in enumerate(x_test_adversarial):
  if (y_test_adversarial[i] == class_pred_clean[i]) and (y_test_adversarial[i] != class_pred_cw[i]):
    data_clean = ([v], None)
    data_adv = ([x_test_carlini[i]], None)
    explanation_clean = gradcam_explainer.explain(data_clean, model_cnn, class_index = y_test_adversarial[i],colormap=cv2.COLORMAP_VIRIDIS)
    explanation_cw = gradcam_explainer.explain(data_adv, model_cnn, class_index = y_test_adversarial[i],colormap=cv2.COLORMAP_VIRIDIS)
    
    axs[0, a].imshow(explanation_clean, cmap='binary')
    axs[0, a].text(15, 32, 'True Class: '+ str(y_test_adversarial[i]), ha='center')
    axs[0, a].text(15, 35, 'Predicted Class: '+ str(class_pred_clean[i]), ha='center')
    axs[0, a].text(15, 38, 'Probability: '+ str(probability_pred_clean[i]), ha='center')
    axs[1, a].imshow(explanation_cw, cmap='binary')
    axs[1, a].text(15, 32, 'True Class: '+ str(y_test_adversarial[i]), ha='center')
    axs[1, a].text(15, 35, 'Predicted Class: '+ str(class_pred_cw[i]), ha='center')
    axs[1, a].text(15, 38, 'Probability: '+ str(probability_pred_cw[i]), ha='center')
    axs[0, a].axis('off')
    axs[1, a].axis('off')
    a += 1
    print(i)
  else:
    i += 1
    v += 1

#Visualize Clean, Perturbed Added, and Adversarial Images on Successfull Attack

f, axs = plt.subplots(2, 5, figsize=(16, 10))
title = ['Clean Images', 'Carlini-Wagner Images']

for row, ax in enumerate(axs, start=0):
    ax[0].set_title("%s \n" % title[row], loc='left', fontsize=14, pad = 0)
    
a = 0

for i,v in enumerate(x_test_adversarial):
  if (y_test_adversarial[i] == class_pred_clean[i]) and (y_test_adversarial[i] != class_pred_cw[i]):
    axs[0, a].imshow( NormalizeData(v), cmap='binary')
    axs[1, a].imshow( NormalizeData(x_test_carlini[i]), cmap='binary')
    axs[0, a].axis('off')
    axs[1, a].axis('off')
    a += 1
  else:
    i += 1
    v += 1

"""## JSMA"""

jsma_attack = SaliencyMapMethod(classifier = cnn_classifier, theta=0.1, gamma=0.01)

start = time.time()
x_test_jsma = jsma_attack.generate(x = x_test_adversarial)
end = time.time()

print(f"Runtime for generating adv images with JSMA is {end - start} second")

#Save generated images into pickle file
with open('/content/gdrive/MyDrive/GTSRB/pickles/x_test_jsma_model2.pk', 'wb') as handle:
    pickle.dump(x_test_jsma, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Predict model accuracy on clean test images
pred_yall = np.argmax(model_cnn.predict(x_all), axis=-1)
accuracy_score(y_all, pred_yall)

start = time.time()
x_test_jsma = jsma_attack.generate(x = x_all)
end = time.time()

print(f"Runtime for generating adv images for training with JSMA is {end - start} second")

#Save generated images into pickle file
with open('/content/gdrive/MyDrive/GTSRB/99accuracy/x_all_jsma.pk', 'wb') as handle:
    pickle.dump(x_test_jsma, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Load pickled JSMA images
with open(r"/content/gdrive/MyDrive/GTSRB/pickles/for model 2/x_test_jsma_model2_gamma.01.pk", "rb") as input_file:
  x_test_jsma = pickle.load(input_file)

# Predict model accuracy on JSMA test images
pred_jsma = np.argmax(model_cnn.predict(x_test_jsma), axis=-1)
accuracy_score(y_test, pred_jsma)
#0.5796516231195566
#0.5901821060965954

#Predict model accuracy on JSMA images for model 1
#Load pickled JSMA images for model 1
with open(r"/content/gdrive/MyDrive/GTSRB/pickles/for model 1/x_test_jsma_model1_gamma.01.pk", "rb") as input_file:
  x_test_jsma_model1 = pickle.load(input_file)

# Predict model accuracy on JSMA test images for model 1
pred_jsma_model1 = np.argmax(model_cnn.predict(x_test_jsma_model1), axis=-1)
accuracy_score(y_test, pred_jsma_model1)

#Predict model accuracy on JSMA images for model 3
#Load pickled JSMA images for model 3
with open(r"/content/gdrive/MyDrive/GTSRB/pickles/for model 3/x_test_jsma_model3_gamma.01.pk", "rb") as input_file:
  x_test_jsma_model3 = pickle.load(input_file)

# Predict model accuracy on JSMA test images for model 3
pred_jsma_model3 = np.argmax(model_cnn.predict(x_test_jsma_model3), axis=-1)
accuracy_score(y_test, pred_jsma_model3)

#Calculate average L2 distance between clean vs JSMA images
sum = 0
for i in range(len(x_test_adversarial)):
  sum += calculateDistance(x_test_adversarial[i], x_test_jsma[i])

avgL2_jsma = sum / len(x_test_adversarial)
print("Avg L2 distance of generated JSMA images: ", avgL2_jsma)
#*Avg L2 distance of generated JSMA images:  9.868329471193979

#Predict Standard CNN Accuracy on JSMA Images
y_pred_jsma = model_cnn(x_test_jsma)
y_pred_jsma = np.argmax(y_pred_jsma, axis=1)
acc = sum(1 for x,y in zip(y_pred_jsma, y_test_adversarial) if x == y) / float(len(y_test_adversarial))
acc

#Get Model Prediction on Adversarial Images
prediction_jsma = model_cnn.predict(x_test_jsma)

probability_pred_jsma = np.max(prediction_jsma, axis=1)
class_pred_jsma = np.argmax(prediction_jsma, axis=1)

#Visualize successful attacks

f, axs = plt.subplots(2, 5, figsize=(15, 8))
title = ['Clean Images', 'JSMA Images']

for row, ax in enumerate(axs, start=0):
    ax[0].set_title("%s \n" % title[row], loc='left', fontsize=16)
  
a=0
for i,v in enumerate(x_test_adversarial):
  if (y_test_adversarial[i] == class_pred_clean[i]) and (y_test_adversarial[i] != class_pred_jsma[i]):
    data_clean = ([v], None)
    data_adv = ([x_test_jsma[i]], None)
    explanation_clean = gradcam_explainer.explain(data_clean, model_cnn, class_index = y_test_adversarial[i],colormap=cv2.COLORMAP_VIRIDIS)
    explanation_jsma = gradcam_explainer.explain(data_adv, model_cnn, class_index = y_test_adversarial[i],colormap=cv2.COLORMAP_VIRIDIS)
    
    axs[0, a].imshow(explanation_clean, cmap='binary')
    axs[0, a].text(15, 32, 'True Class: '+ str(y_test_adversarial[i]), ha='center')
    axs[0, a].text(15, 35, 'Predicted Class: '+ str(class_pred_clean[i]), ha='center')
    axs[0, a].text(15, 38, 'Probability: '+ str(probability_pred_clean[i]), ha='center')
    axs[1, a].imshow(explanation_jsma, cmap='binary')
    axs[1, a].text(15, 32, 'True Class: '+ str(y_test_adversarial[i]), ha='center')
    axs[1, a].text(15, 35, 'Predicted Class: '+ str(class_pred_jsma[i]), ha='center')
    axs[1, a].text(15, 38, 'Probability: '+ str(probability_pred_jsma[i]), ha='center')
    axs[0, a].axis('off')
    axs[1, a].axis('off')
    a += 1
  else:
    i += 1
    v += 1

#Visualize Clean, Perturbed Added, and Adversarial Images on Successfull Attack

f, axs = plt.subplots(2, 5, figsize=(16, 10))
title = ['Clean Images', 'JSMA Images']

for row, ax in enumerate(axs, start=0):
    ax[0].set_title("%s \n" % title[row], loc='left', fontsize=14, pad = 0)
    
a = 0

for i,v in enumerate(x_test_adversarial):
  if (y_test_adversarial[i] == class_pred_clean[i]) and (y_test_adversarial[i] != class_pred_jsma[i]):
    
    axs[0, a].imshow(v, cmap='binary')
    axs[1, a].imshow(x_test_jsma[i], cmap='binary')
    axs[0, a].axis('off')
    axs[1, a].axis('off')
    a += 1
    print(i)
  else:
    i += 1
    v += 1
	
"""## DeepFool"""

df_attack = DeepFool(classifier= cnn_classifier)

start = time.time()
x_test_df = df_attack.generate(x=x_test_adversarial)
end = time.time()

print(f"Runtime for generating adv images with DeepFool is {end - start} second")

#Save generated images into pickle file
with open('/content/gdrive/MyDrive/GTSRB/pickles/x_test_df_model2.pk', 'wb') as handle:
    pickle.dump(x_test_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Load pickled DeepFool images
with open(r"/content/gdrive/MyDrive/GTSRB/pickles/for model 2/x_test_df_model2.pk", "rb") as input_file:
  x_test_df = pickle.load(input_file)

# Predict model accuracy on DeepFool test images
pred_df = np.argmax(model_cnn.predict(x_test_df), axis=-1)
accuracy_score(y_test, pred_df)
#*0.7880443388756928

#Predict model accuracy on DF images for model 2
#Load pickled DF images for model 2
with open(r"/content/gdrive/MyDrive/GTSRB/pickles/for model 3/x_test_df_model3.pk", "rb") as input_file:
  x_test_df_model3 = pickle.load(input_file)

# Predict model accuracy on JSMA test images for model 2
pred_df_model3 = np.argmax(model_cnn.predict(x_test_df_model3), axis=-1)
accuracy_score(y_test, pred_df_model3)

#Predict model accuracy on DF images for model 1
#Load pickled DF images for model 1
with open(r"/content/gdrive/MyDrive/GTSRB/pickles/for model 1/x_test_df_model1.pk", "rb") as input_file:
  x_test_df_model1 = pickle.load(input_file)

# Predict model accuracy on JSMA test images for model 1
pred_df_model1 = np.argmax(model_cnn.predict(x_test_df_model1), axis=-1)
accuracy_score(y_test, pred_df_model1)

#Calculate average L2 distance between clean vs DeepFool images
sum = 0
for i in range(len(x_test_adversarial)):
  sum += calculateDistance(x_test_adversarial[i], x_test_df[i])

avgL2_df = sum / len(X_test)
print("Avg L2 distance of generated DeepFool images: ", avgL2_df)
#Avg L2 distance of generated DeepFool images:  0.7909094764388509

#Predict Standard CNN Accuracy on Adversarial Images Deep Fool
y_pred_df = model_cnn(x_test_df)
y_pred_df = np.argmax(y_pred_df, axis=1)
acc = sum(1 for x,y in zip(y_pred_df, y_test_adversarial) if x == y) / float(len(y_test_adversarial))
acc
#0.6995249406175772

#Get Model Prediction on Adversarial Images
prediction_df = model_cnn.predict(x_test_df)

probability_pred_df = np.max(prediction_df, axis=1)
class_pred_df = np.argmax(prediction_df, axis=1)
#Visualize successful attacks

f, axs = plt.subplots(2, 5, figsize=(15, 8))
title = ['Clean Images', 'DeepFool Images']

for row, ax in enumerate(axs, start=0):
    ax[0].set_title("%s \n" % title[row], loc='left', fontsize=16)
  
a=0
for i,v in enumerate(x_test_adversarial):
  if (y_test_adversarial[i] == class_pred_clean[i]) and (y_test_adversarial[i] != class_pred_df[i]):
    data_clean = ([v], None)
    data_adv = ([x_test_df[i]], None)
    explanation_clean = gradcam_explainer.explain(data_clean, model_cnn, class_index = y_test_adversarial[i],colormap=cv2.COLORMAP_VIRIDIS)
    explanation_df = gradcam_explainer.explain(data_adv, model_cnn, class_index = y_test_adversarial[i],colormap=cv2.COLORMAP_VIRIDIS)
    
    axs[0, a].imshow(explanation_clean, cmap='binary')
    axs[0, a].text(15, 32, 'True Class: '+ str(y_test_adversarial[i]), ha='center')
    axs[0, a].text(15, 35, 'Predicted Class: '+ str(class_pred_clean[i]), ha='center')
    axs[0, a].text(15, 38, 'Probability: '+ str(probability_pred_clean[i]), ha='center')
    axs[1, a].imshow(explanation_df, cmap='binary')
    axs[1, a].text(15, 32, 'True Class: '+ str(y_test_adversarial[i]), ha='center')
    axs[1, a].text(15, 35, 'Predicted Class: '+ str(class_pred_df[i]), ha='center')
    axs[1, a].text(15, 38, 'Probability: '+ str(probability_pred_df[i]), ha='center')
    axs[0, a].axis('off')
    axs[1, a].axis('off')
    a += 1
  else:
    i += 1
    v += 1

#Visualize Clean, Perturbed Added, and Adversarial Images on Successfull Attack

f, axs = plt.subplots(3, 5, figsize=(16, 10))
title = ['Clean Images', 'Perturbation Added', 'DeepFool Images']

for row, ax in enumerate(axs, start=0):
    ax[0].set_title("%s \n" % title[row], loc='left', fontsize=14, pad = 0)
    
a = 0

for i,v in enumerate(x_test_adversarial):
  if (y_test_adversarial[i] == class_pred_clean[i]) and (y_test_adversarial[i] != class_pred_df[i]):
    
    axs[0, a].imshow(v, cmap='binary')
    axs[1, a].imshow(NormalizeData(v)-NormalizeData(x_test_df[i]), cmap='binary')
    axs[2, a].imshow(x_test_df[i], cmap='binary')
    axs[0, a].axis('off')
    axs[1, a].axis('off')
    axs[2, a].axis('off')
    a += 1
    print(i)
  else:
    i += 1
    v += 1