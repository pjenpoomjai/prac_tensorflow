import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

class_names = ['cat','dog']
dirname = os.path.dirname(__file__)
video_name = 'a.mp4'
video_path = os.path.join(dirname,'images', video_name)

with tf.Session() as sess:
#     model= keras.Sequential([
#     keras.layers.Convolution2D(filters=32,kernel_size=2,activation="relu",input_shape=(28,28,3)),
#     keras.layers.MaxPooling2D(pool_size=2),
#     keras.layers.Dropout(0.5),

#     keras.layers.Convolution2D(filters=128,kernel_size=2,activation="relu"),
#     keras.layers.MaxPooling2D(pool_size=2),
#     keras.layers.Dropout(0.5),

#     keras.layers.Convolution2D(filters=256,kernel_size=2,activation="relu"),
#     keras.layers.MaxPooling2D(pool_size=2),
#     keras.layers.Dropout(0.5),

#     keras.layers.Flatten(),
#     keras.layers.Dense(64,activation="relu"),
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
    model.load_weights('./model/cp-0080.ckpt')
    video_capture = cv2.VideoCapture(video_path)

    for i in os.listdir(os.path.join(dirname,'images')):
        if i.endswith('png') or i.endswith('jpg'):
            c = cv2.imread(os.path.join(dirname,'images',i),1)
            c = cv2.resize(c,(28,28))
	    c = np.array(c)
            newFrame = np.array([c]) /255.0
            predictions = model.predict(newFrame)
            print(class_names[np.argmax(predictions)])


	# #frameRate = video_capture.get(5) #frame rate
    # while True:  # fps._numFrames < 120
    #     ret,frameOriginal = video_capture.read() # get current frame
    #     frame = cv2.cvtColor(frameOriginal, 1)
    #     print(frame.shape)
    #     newFrame = cv2.resize(frame,(28,28))
    #     print(newFrame.shape)
    #     newFrame = np.array(newFrame)
    #     newFrame = np.array([newFrame]) /255.0
    #     predictions = model.predict(newFrame)
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     if np.max(predictions) > 0.9:
    #         animal = class_names[np.argmax(predictions)]
    #         cv2.putText(frameOriginal,"%s"%animal,(10,500), font, 1,(0,255,0),2)
    #     cv2.imshow('result',frameOriginal)
    #     print(predictions[0][0],predictions[0][1])
    #     cv2.waitKey(30) 
    # video_capture.release() # handle it nicely
    # cv2.destroyAllWindows() # muahahaha
