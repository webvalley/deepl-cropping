import os
import numpy as np
import sklearn
import pydicom
import matplotlib.pyplot as plt
import pickle

np.random.seed(42)

from model import *
from data import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dimension = (576, 640, 1)

def createMask(file):
    for key in file.dir():
       value = getattr(file, key, "")
       if(key == "SequenceOfUltrasoundRegions"):
           value = value[0]
           break
    x0, x1, y0, y1 = None, None, None, None
    for key in value.dir():
       if key == "RegionLocationMinX0":
           x0 = getattr(value, key, "")
       if key == "RegionLocationMaxX1":
           x1 = getattr(value, key, "")
       if key == "RegionLocationMinY0":
           y0 = getattr(value, key, "")
       if key == "RegionLocationMaxY1":
           y1 = getattr(value, key, "")            
    
    masked = np.zeros(file.pixel_array.shape)
    masked[y0:y1+1, x0:x1+1, 0] = 1
    
    
    return masked[:, :, :1]

folder = "/data3/wv2019/data/PLIC_CHIESA_DICOM/"

x = []
y = []
for filename in os.listdir(folder)[:750]:
    file = pydicom.dcmread(os.path.join(folder, filename))
    if (file.pixel_array[:, :, :1].shape == dimension):
        x.append(file.pixel_array[:, :, :1])
        masked = createMask(file)
        y.append(masked)

X = np.array(x)
Y = np.array(y)

model = unet(input_size=dimension)

model_checkpoint = ModelCheckpoint('unet_check.hdf5', monitor='loss', verbose=1, save_best_only=True)

history = model.fit(X, Y, batch_size=2, epochs=500, verbose=1, 
                    validation_split=0.33, shuffle=True, 
                    callbacks=[model_checkpoint])


model.save("model_500ep.h5")
pickle.dump(history, open("model_500ep.pkl", "wb"))