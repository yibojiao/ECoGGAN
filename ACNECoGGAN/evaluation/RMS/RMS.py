#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 09:53:48 2021

@author: praneethsv
"""
import pickle
import numpy as np
from PIL import Image
import os

#%%
result_data = '/home/kaseya/Downloads/gan-toolkit-pattern_v4/agant/evaluation/imagesFID/dataset2'

train_data = '/home/kaseya/Downloads/gan-toolkit-pattern_v4/agant/evaluation/imagesFID/dataset1'
result_imgs = np.sort(os.listdir(result_data))
train_imgs = np.sort(os.listdir(train_data))

image_result = []
for file in result_imgs:
    img = Image.open(result_data+'/'+file)
    img = img.convert('L')
    img_arr = np.array(img)
    img = np.asarray(img_arr)/255
    image_result.append(img)  
image_result = np.array(image_result)

image_train = []
for file in train_imgs:
    img = Image.open(train_data+'/'+file)
    img = img.convert('L')
    img_arr = np.array(img)
    img = np.asarray(img_arr)/255
    image_train.append(img)  
image_train = np.array(image_train)

#%%
at,bt,ct = image_train.shape
ar,br,cr = image_result.shape
score_list = []

for a in range(at):
    for c in range(ct):
        train_test = image_train[a,:,c]
        rms_signal = ((train_test) ** 2).mean() ** 0.5
        cha_list = []
        
        for d in range(ar):
            result_test = image_result[d,:,c]
            rms_noise = ((train_test - result_test) ** 2).mean() ** 0.5             
            score = 1 - rms_noise / rms_signal
            cha_list.append(score)
        cha_array = np.array(cha_list)
        score_list.append(cha_array)
#%%
score_array = np.asarray(score_list)
sample_ave = []
for ay in score_array:
    ave = np.mean(ay)
    sample_ave.append(ave)
all_ave = np.mean(sample_ave)
print(all_ave)
