import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2
import os

def read_dataset(datasetPath):
    cache_path = datasetPath.rstrip('/') + '.cache'
    print('\nLoading dataset list from cache...\n')
    with open(cache_path, 'rb') as f:
        trainImages,trainLabels,testImages,testLabels = pickle.load(f)
    return trainImages,trainLabels,testImages,testLabels


def build_dataset(trainImages,trainLabels,testImages,testLabels,datasetPath):
    cache_path = datasetPath.rstrip('/') + '.cache'
    print('\nEnumerating dataset list from disk...\n')
    with open(cache_path, 'wb') as f:
        pickle.dump((trainImages,trainLabels,testImages,testLabels), f)

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'images')
catPath = os.path.join(filename, 'cat')
dogPath = os.path.join(filename, 'dog')
trainImages = []
testImages = []
trainLabels = [] 
testLabels = []
class_names = ['cat','dog']

cache = 1
datasetPath = 'dataset'
if cache == 1:
    trainImages,trainLabels,testImages,testLabels = read_dataset(datasetPath)
else:
    #cat
    number = 0
    for i in os.listdir(catPath):
        if i.endswith('.png') or i.endswith('.jpg'):
            c = cv2.imread(catPath+'/'+i,1)
            print(c.shape)
            c = cv2.resize(c,(28,28))
            c = np.array(c)
            print(c.shape)
            cv2.imshow('cat',c)
            cv2.waitKey(500)
            
            if number <120:
                trainImages.append(c)
                trainLabels.append(0)
            else:
                
                testImages.append(c)
                testLabels.append(0)
            number+=1    
    #dog
    number = 0
    for i in os.listdir(dogPath):
        if i.endswith('.png') or i.endswith('.jpg'):
            c = cv2.imread(dogPath+'/'+i,1)
            print(c.shape)
            c = cv2.resize(c,(28,28))
            c = np.array(c)
            print(c.shape)
            cv2.imshow('dog',c)
            cv2.waitKey(500)
            if number <120:
                trainImages.append(c)
                trainLabels.append(1)
            else:
                testImages.append(c)
                testLabels.append(1)
            number+=1    
    build_dataset(trainImages,trainLabels,testImages,testLabels,datasetPath)


trainImages = np.array(trainImages)        
trainLabels = np.array(trainLabels)        
testI = testImages
testImages = np.array(testImages)
testLabels = np.array(testLabels)

trainImages = trainImages / 255.0
testImages = testImages / 255.0

# model= keras.Sequential([
#     keras.layers.Convolution2D(filters=32,kernel_size=3,activation="relu",input_shape=(28,28,3)),
#     keras.layers.Convolution2D(filters=32,kernel_size=3,activation="relu"),
#     keras.layers.MaxPooling2D(pool_size=2),
#     keras.layers.Dropout(0.5),

#     # keras.layers.Convolution2D(filters=64,kernel_size=3,activation="relu"),
#     # keras.layers.Convolution2D(filters=64,kernel_size=3,activation="relu"),
#     # keras.layers.MaxPooling2D(pool_size=2),
#     # keras.layers.Dropout(0.5),

#     keras.layers.Convolution2D(filters=64,kernel_size=3,activation="relu"),
#     keras.layers.Convolution2D(filters=64,kernel_size=3,activation="relu"),
#     keras.layers.MaxPooling2D(pool_size=2),
#     keras.layers.Dropout(0.5),

#     keras.layers.Flatten(),
#     keras.layers.Dense(512,activation="relu"),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(2, activation=tf.nn.softmax)
# ])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28,3)),
    keras.layers.Dense(128, activation=tf.nn.relu),    
    keras.layers.Dense(16, activation=tf.nn.relu),    
    keras.layers.Dropout(0.2),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#save model
checkpoint_path = "model/cp-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=10)
model.save_weights(checkpoint_path.format(epoch=0))
# model.load_weights('./model/cp-0300.ckpt')  #initial_epoch=300
model.fit(trainImages, trainLabels, epochs=80,callbacks = [cp_callback])

test_loss, test_acc = model.evaluate(testImages, testLabels)
print('Test accuracy:', test_acc)



