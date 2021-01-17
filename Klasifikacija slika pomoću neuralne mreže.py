# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 15:01:17 2020

@author: Nola Čumlievski
"""

import numpy as np
import tensorflow as tf
import pandas as pd
from emnist import list_datasets
from emnist import extract_training_samples
from emnist import extract_test_samples
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
import random
import seaborn as sns
from cv2 import cv2
from imutils import contours
from PIL import Image, ImageFilter


# importanje skupa podataka
list_datasets()

# izrada skupova za treniranje i testiranje
treniranje, treniranje_oznake = extract_training_samples('balanced')
testiranje, testiranje_oznake = extract_test_samples('balanced')
plt.imshow(treniranje[1].reshape([28, 28]), cmap='Greys_r')
plt.show()
np.unique(treniranje_oznake)

# skaliranje vrijednosti
treniranje_podaci = np.copy(treniranje)
treniranje_podaci = treniranje_podaci.astype('float32')
treniranje_podaci /= 255.

testiranje_podaci = np.copy(testiranje)
testiranje_podaci = testiranje_podaci.astype('float32') #potrebno je pretvoriti u float32
testiranje_podaci /= 255.

for i in range(9):  
    plt.subplot(330 + 1 + i)
    plt.imshow(treniranje_podaci[i], cmap=plt.get_cmap('gray'))
    plt.show()
treniranje_oznake[:9]

# distribucija klasa
plt.figure(figsize=(8,5))
plt.hist(treniranje_oznake, bins=47)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')



""" # reshape podataka ručno ili u funkciji modela - treniranje_podaci1 = treniranje_podaci.reshape(treniranje_podaci.shape[0], 28, 28, 1)
testiranje_podaci1 = testiranje_podaci.reshape(testiranje_podaci.shape[0], 28, 28, 1)"""

# izrada skupa za validaciju
treniranje_podaci, validacija_podaci, treniranje_oznake, validacija_oznake = train_test_split(treniranje_podaci, treniranje_oznake, test_size=0.15, random_state=1)

# izrada DNN mreže

# parametri
input_veličina = 784 # podacisu formata 28 X 28
output_veličina = 47 # broj klasa
hidden_layer_veličina = 700

model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape = (28, 28, 1)),
                             tf.keras.layers.Dense(hidden_layer_veličina, activation = "relu"),
                             tf.keras.layers.Dense(hidden_layer_veličina, activation = "relu"),
                             tf.keras.layers.Dense(output_veličina, activation = "softmax")])

# specifikacija optimizatora
optimizer = keras.optimizers.Adam(lr=0.005)
model.compile(optimizer = optimizer, loss = "sparse_categorical_crossentropy", metrics = "accuracy")

# treniranje modela
broj_epoha =10
np.random.seed(30)
fit = model.fit(treniranje_podaci, treniranje_oznake, batch_size = 5000, epochs = broj_epoha, verbose=2, validation_data=(validacija_podaci, validacija_oznake))

# testiranje modela na skupu za testiranje
rezultati = model.evaluate(testiranje_podaci, testiranje_oznake)

# vizualizacija performansi
plt.figure(figsize=(12, 6), dpi=96)
plt.subplot(1, 2, 1)
plt.plot(fit.history['loss'])
plt.plot(fit.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train', 'test'], loc='upper left')
plt.subplot(1, 2, 2)
plt.plot(fit.history['accuracy'])
plt.plot(fit.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
plt.show()

# matrica konfuzije
rezultati1 = model.predict_classes(testiranje_podaci, verbose=1)
data = {'Prave klase': testiranje_oznake,'Predviđene klase': rezultati1}

df = pd.DataFrame(data, columns=['Prave klase','Predviđene klase'])
confusion_matrix = pd.crosstab(df['Prave klase'], df['Predviđene klase'], rownames=['Prave klase'], colnames=['Predviđene klase'], margins = True)
confusion_matrix = confusion_matrix.drop(index='All', columns='All')
confusion_matrix.columns = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t"]
confusion_matrix.rename(index={0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10:"A", 11:"B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I", 19: "J", 20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 29: "T", 30: "U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z", 36: "a", 37: "b", 38: "d", 39: "e", 40: "f", 41: "g", 42: "h", 43: "n", 44: "q", 45: "r", 46: "t"}, inplace = True)

plt.figure(figsize = (50, 50))  #This is the size of the image
heatM = sns.heatmap(confusion_matrix, linewidths=.7,  center = 0, cmap = sns.diverging_palette(20, 220, n = 60), annot = True, fmt='d') #this are the caracteristics of the heatmap
heatM.set_ylim([47,0]) 

# učitavanje skenirane riječi, mijenjanje boja, riječ marks

image = cv2.imread("marks.png")
cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Output", image)
cv2.waitKey(0)

invert = cv2.bitwise_not(image)
cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Output", invert)
cv2.waitKey(0)

# podjela slike na pojedina slova

gray = cv2.cvtColor(invert, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts, _ = contours.sort_contours(cnts, method="left-to-right")

ROI_number = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > 10:
        x,y,w,h = cv2.boundingRect(c)
        ROI = 255 - image[y:y+h, x:x+w]
        cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        cv2.rectangle(invert, (x, y), (x + w, y + h), (36,255,12), 1)
        ROI_number += 1
cv2.imshow('thresh', invert)
cv2.waitKey()

# convert slike u 28 X 28

# definicija funkcije za convert slike da bi se mreža mogla testirati na njoj
def imageprepare(argv): #Iinput je file path od slike
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (0))  # kreira canvas od 28 X 28

    if width > height:  #provjera koja je dimenzija duža
        nheight = int(round((20.0 / width * height), 0)) 
        if (nheight == 0):  
            nheight = 1
            # resize i oštrina
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # kalkulacija horizontalne širine
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        nwidth = int(round((20.0 / height * width), 0))  # resize dužine prema visini
        if (nwidth == 0):
            nwidth = 1
            # resize i oštrina
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # kalkulacija vertikalne pozicije
        newImage.paste(img, (wleft, 4))  

    tv = list(newImage.getdata())  # vrijednosti pixela

    # normalizacija pixela
    tva = [(x / 255.) for x in tv]
    print(tva)
    return tva

x=imageprepare('ROI_0.png')
# treba biti dužina 784
x = np.array(x)
plt.imshow(x.reshape([28, 28]), cmap='Greys_r')
plt.show()

# predviđanje
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

x=imageprepare('ROI_1.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

x=imageprepare('ROI_2.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

x=imageprepare('ROI_3.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

x=imageprepare('ROI_4.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat)

# questions

image = cv2.imread("questions.png")
invert = cv2.bitwise_not(image)

gray = cv2.cvtColor(invert, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts, _ = contours.sort_contours(cnts, method="left-to-right")

ROI_number = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > 10:
        x,y,w,h = cv2.boundingRect(c)
        ROI = 255 - image[y:y+h, x:x+w]
        cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 1)
        ROI_number += 1
cv2.imshow('thresh', invert)
cv2.waitKey()

x=imageprepare('ROI_0.png')
x = np.array(x)
# predviđanje - questions
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) 

x=imageprepare('ROI_1.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) # točno!

x=imageprepare('ROI_2.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

x=imageprepare('ROI_3.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

x=imageprepare('ROI_4.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) # opet t prevodi kao I!

x=imageprepare('ROI_5.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) 

x=imageprepare('ROI_7.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

x=imageprepare('ROI_8.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat)  #točno!

x=imageprepare('ROI_9.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

#riječ company
image = cv2.imread('company.png')
invert = cv2.bitwise_not(image) #jako važno, inače algoritam slabo prepoznaje slova na biloj pozadini

gray = cv2.cvtColor(invert, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts, _ = contours.sort_contours(cnts, method="left-to-right")

ROI_number = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > 10:
        x,y,w,h = cv2.boundingRect(c)
        ROI = 255 - image[y:y+h, x:x+w]
        cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 1)
        ROI_number += 1
cv2.imshow('thresh', invert)
cv2.waitKey()

x=imageprepare('ROI_0.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) 

x=imageprepare('ROI_1.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

x=imageprepare('ROI_2.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

x=imageprepare('ROI_3.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

x=imageprepare('ROI_4.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

x=imageprepare('ROI_5.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat)  #točno!

x=imageprepare('ROI_7.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat)  #točno!

# ručna slika - overmatching
image = cv2.imread('image.png')
cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Output", image)
cv2.waitKey(0)

invert = cv2.bitwise_not(image)
cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Output", invert)
cv2.waitKey(0)
 #jako važno, inače algoritam slabo prepoznaje slova na biloj pozadini

gray = cv2.cvtColor(invert, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts, _ = contours.sort_contours(cnts, method="left-to-right")

ROI_number = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > 10:
        x,y,w,h = cv2.boundingRect(c)
        ROI = 255 - image[y:y+h, x:x+w]
        cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 1)
        ROI_number += 1
cv2.imshow('thresh', image)

# 0
x=imageprepare('ROI_0.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

#v
x=imageprepare('ROI_1.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

# e
x=imageprepare('ROI_2.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

# r
x=imageprepare('ROI_3.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

# m
x=imageprepare('ROI_4.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #netočno!

# a
x=imageprepare('ROI_5.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

# t
x=imageprepare('ROI_6.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!!

# c
x=imageprepare('ROI_7.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

# h 
x=imageprepare('ROI_8.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

# i
x=imageprepare('ROI_9.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno

# n
x=imageprepare('ROI_11.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno

# g
x=imageprepare('ROI_12.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #netočno

# druga ručno napisana rijec - transfix
image = cv2.imread('transfix.png')
cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Output", image)
cv2.waitKey(0)

invert = cv2.bitwise_not(image)
cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Output", invert)
cv2.waitKey(0)

gray = cv2.cvtColor(invert, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts, _ = contours.sort_contours(cnts, method="left-to-right")

ROI_number = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > 10:
        x,y,w,h = cv2.boundingRect(c)
        ROI = 255 - image[y:y+h, x:x+w]
        cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 1)
        ROI_number += 1
cv2.imshow('thresh', image)

# slovo T
x=imageprepare('ROI_0.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

# slovo R
x=imageprepare('ROI_1.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

# slovo A
x=imageprepare('ROI_2.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #tiočno!

# slovo N
x=imageprepare('ROI_3.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno

# S
x=imageprepare('ROI_4.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

# malo f
x=imageprepare('ROI_5.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) # registrirano malo f kao veliko

# malo i
x=imageprepare('ROI_7.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

# malo x
x=imageprepare('ROI_8.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat)

# treći primjer - judge

image = cv2.imread('judgemental.png')
cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Output", image)
cv2.waitKey(0)

invert = cv2.bitwise_not(image)
cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Output", invert)
cv2.waitKey(0)

gray = cv2.cvtColor(invert, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts, _ = contours.sort_contours(cnts, method="left-to-right")

ROI_number = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > 10:
        x,y,w,h = cv2.boundingRect(c)
        ROI = 255 - image[y:y+h, x:x+w]
        cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 1)
        ROI_number += 1
cv2.imshow('thresh', image)

# J
x=imageprepare('ROI_0.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #bravo!

# u
x=imageprepare('ROI_2.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno

# d
x=imageprepare('ROI_3.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno

# D
x=imageprepare('ROI_4.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat)

# g
x=imageprepare('ROI_5.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat)

# G
x=imageprepare('ROI_6.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat)

# m
x=imageprepare('ROI_7.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno

# e
x=imageprepare('ROI_8.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat)  #točno

# n
x=imageprepare('ROI_9.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno

# t
x=imageprepare('ROI_10.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno

# a
x=imageprepare('ROI_11.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

# L
x=imageprepare('ROI_13.png')
x = np.array(x)
rezultat = model.predict_classes(x.reshape((1, 28, 28)), verbose=1)
print(rezultat) #točno!

9/12 * 100












