# '''
from utils import *
import numpy as np
import pandas as pd
import cv2
from pixiv_grabber import PixivDataset
from configs import *



img = cv2.imread('C:/Users/quale/Desktop/New folder (8)/koishi.png', cv2.IMREAD_COLOR)
cv2.imshow('', img); cv2.waitKey(0)
img = np.expand_dims(cv2.resize(img, (299, 299)), 0)
img = img.astype(np.float32)
img = img/255

import tensorflow as tf
model = tf.keras.models.load_model(REDDIT_MODEL_DIR)

pred = model(img, training=False)[0]
thclist = THClist(all_characters, all_sisters)
print(pred)
print(thclist.one_hot_decode(pred))

grabber = PixivDataset(batches=1)
print(grabber.test_ds.batches)
# for X, y in grabber.test_ds:
#     pred = model(X, training=False)
#
#     print('Prediction:', grabber.one_hot_decode(pred[0], top=5))
#
#     print('Value:', grabber.one_hot_decode(y[0], top=None))
#     cv2.imshow('', cv2.resize(X[0], None, fx=3.0, fy=3.0))
#     cv2.waitKey(0)

# '''

