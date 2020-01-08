import tensorflow as tf
import keras
import os
from scipy import misc
import numpy as np
import random

model = keras.applications.MobileNetV2(input_shape=(66, 200, 4), weights=None, classes=1)
SGD = keras.optimizers.SGD(lr=0.01, decay=1e-6)
model.compile(optimizer='SGD', loss='mean_squared_error', metrics=['accuracy'])

imgs0 = []
imgs1 = []

batchSize = 50

for file in os.listdir("0"):
    image = misc.imread("0/{}".format(file))
    # print(image.shape)
    imgs0.append(image)

for fi in os.listdir("1"):
    image = misc.imread("1/{}".format(fi))
    imgs1.append(image)

TrainData = []
for a in range(len(imgs0)):
    TrainData.append([imgs0[a], 0])

for b in range(len(imgs1)):
    TrainData.append([imgs1[b], 1])

random.shuffle(TrainData)
inputX, inputY = [(t[0], t[1]) for t in TrainData]
model.fit(X=inputX, Y=inputY, batchSize=50)
