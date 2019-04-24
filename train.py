import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
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
        print(i)
        if i.endswith('.png') or i.endswith('.jpg'):
            c = cv2.imread(catPath+'/'+i,0)
            print(c.shape)
            c = cv2.resize(c,(300,300))
            print(c.shape)
            cv2.imshow('cat',c)
            cv2.waitKey(500)
            
            if number <130:
                trainImages.append(c)
                trainLabels.append(0)
            else:
                
                testImages.append(c)
                testLabels.append(0)
            number+=1    
    #dog
    number = 0
    for i in os.listdir(dogPath):
        print(i)
        if i.endswith('.png') or i.endswith('.jpg'):
            c = cv2.imread(dogPath+'/'+i,0)
            print(c.shape)
            c = cv2.resize(c,(300,300))
            print(c.shape)
            cv2.imshow('dog',c)
            cv2.waitKey(500)
            if number <130:
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




model = keras.Sequential([
    keras.layers.Flatten(input_shape=(300, 300)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(32, activation=tf.nn.relu),
    # keras.layers.Dropout(0.25),
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
    period=5)
model.save_weights(checkpoint_path.format(epoch=0))
# model.load_weights('./model/cp-0300.ckpt')  #initial_epoch=300
model.fit(trainImages, trainLabels, epochs=500,callbacks = [cp_callback])

test_loss, test_acc = model.evaluate(testImages, testLabels)
print('Test accuracy:', test_acc)



# predictions = model.predict(testImages)
# model.load_weights('./model/cp-0040.ckpt')
# p =0
# for i in predictions:
#     cv2.imshow('result',testImages[p])
#     cv2.waitKey(2000)
#     print('predic :',class_names[np.argmax(i)],'(True',testLabels[p],')')
#     p+=1




