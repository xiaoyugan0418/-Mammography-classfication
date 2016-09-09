################
# benign = 0 ; Cancer = 1; Normal = 2
#
#
#################################
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib import urlretrieve
import cPickle as pickle
import os
import gzip
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import glob
import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split

# def load_dataset():
#
#     filename = 'mnist.pkl.gz'
#     with gzip.open(filename, 'rb') as f:
#         data = pickle.load(f)
#     X_train, y_train = data[0]
#     X_val, y_val = data[1]
#     X_test, y_test = data[2]
#     X_train = X_train.reshape((-1, 1, 28, 28))
#     X_val = X_val.reshape((-1, 1, 28, 28))
#     X_test = X_test.reshape((-1, 1, 28, 28))
#     y_train = y_train.astype(np.uint8)
#     y_val = y_val.astype(np.uint8)
#     y_test = y_test.astype(np.uint8)
#     return X_train, y_train, X_val, y_val, X_test, y_test
#
# X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
#
# print(X_train);
# print("x0")
# print(X_train[0])
# img = cv2.imread('0.png',0)
# img = img.reshape(-1,1,256,256)
# print("imagedata")
# print(img)

X = []
y=[]
filename_benign = ["/benigns/%d.png" % r for r in range(0, 103)]
filename_cancers = ["/cancers/%d.png" % r for r in range(0, 178)]

for fn in filename_benign:
    im = cv2.imread(str(fn),0)
    im1=im.reshape(-1, 1, 256, 256)
    X.append(im1/255.0)
    y.append(0)

for fn in filename_cancers:
    im = cv2.imread(str(fn),0)
    im1=im.reshape(-1, 1, 256, 256)
    X.append(im1/255.0)
    y.append(1)

mypath='/ROIs/normals/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for n in range(0, len(onlyfiles)):
  images = cv2.imread(join(mypath,onlyfiles[n]), 0)
  im1 = images.reshape(-1, 1, 256, 256)
  X.append(im1/255.0)
  y.append(2)


X = np.array(X).astype(np.float32)
y = np.array(y).astype(np.int32)
print y
X = X.reshape(-1,1,256,256)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# print X_train[0]
# print X_train[500].shape
# print y_train
# print X_test
# print y_test
net1 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('conv2d3', layers.Conv2DLayer),
            ('maxpool3', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    # input layer
    input_shape=(None, 1, 256, 256),
    # layer conv2d1
    conv2d1_num_filters=32,
    conv2d1_filter_size=(3, 3),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),
    # layer maxpool1
    maxpool1_pool_size=(2, 2),
    # layer conv2d2
    conv2d2_num_filters=64,
    conv2d2_filter_size=(3, 3),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2
    maxpool2_pool_size=(2, 2),
    # layer conv2d2
    conv2d3_num_filters=128,
    conv2d3_filter_size=(3, 3),
    conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2
    maxpool3_pool_size=(2, 2),
    # dropout1
    dropout1_p=0.5,
    # dense
    dense_num_units=256,
    dense_nonlinearity=lasagne.nonlinearities.rectify,
    # dropout2
    dropout2_p=0.5,
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=3,
    # optimization method params
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=100,
    verbose=1,
    )
nn = net1.fit(X_train,y_train)

import cPickle as pickle
with open('net1.pickle', 'wb') as f:
    pickle.dump(net1, f, -1)


preds = net1.predict(X_test)
cm = confusion_matrix(y_test, preds)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


train_loss = np.array([i["train_loss"] for i in net1.train_history_])
valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
plt.plot(train_loss, linewidth=3, label="train")
plt.plot(valid_loss, linewidth=3, label="valid")
plt.grid()
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.ylim(1e-3, 1e-2)
plt.show()


