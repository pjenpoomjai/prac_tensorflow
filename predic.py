import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

class_names = ['cat','dog']
dirname = os.path.dirname(__file__)
video_name = 'FUNNY.mp4'
video_path = os.path.join(dirname, video_name)

with tf.Session() as sess:
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(32, activation=tf.nn.relu),
        # keras.layers.Dropout(0.25),
        keras.layers.Dense(2, activation=tf.nn.softmax)
        ])
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.load_weights('./model/cp-0100.ckpt')
    video_capture = cv2.VideoCapture(video_path) 
	#frameRate = video_capture.get(5) #frame rate
    while True:  # fps._numFrames < 120
        ret,frameOriginal = video_capture.read() # get current frame
        frame = cv2.cvtColor(frameOriginal, cv2.COLOR_BGR2GRAY)
        # frame = cv2.imread(frame,0)
        print(frame.shape)
        newFrame = cv2.resize(frame,(28,28))
        newFrame = np.reshape(newFrame,[1,28,28])
        print(newFrame.shape)
        # frameId = video_capture.get(1) #current frame number
        # newFrame = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY)
        cv2.waitKey(1)
        # npNewFrame = (np.array(newFrame) / 255.0)
        # print(npNewFrame.shape)
        predictions = model.predict(newFrame)
		#if (frameId % math.floor(frameRate) == 0):
        if (0 == 0):  # not necessary
            cv2.imshow('result',frameOriginal)
            print(predictions)
            if np.argmax(predictions) ==0:
                print("CAT")
                cv2.waitKey(5000)
            # print('predic :',class_names[np.argmax(predictions)])
            cv2.waitKey(1)  # wait 1ms -> 0 until key input
    video_capture.release() # handle it nicely
    cv2.destroyAllWindows() # muahahaha