from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from imutils import paths
import numpy as np
from collections import defaultdict
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import cv2
import numpy as np
import pandas as pd
import sys
from tkinter import ttk
import os
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from playsound import playsound

main = tkinter.Tk()
main.title("PERSONALISED MUSIC RECOMMENDATION SYSTEM")
main.geometry("1000x500")

global value
global filename
global faces
global frame
detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = '_mini_XCEPTION.106-0.65.hdf5'
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]

def song_prediction(emotion):
    emotions = {"angry" : 1, "disgust" : 2, "happy" : 3, "neutral" : 4, "sad" : 5, "scared" : 6, "surprise" : 7}
    x_new = emotions[emotion]
    dataset=pd.read_csv("Songs_Data.csv")
    dataset1=pd.read_csv("Songs_Data.csv")
    X_train = dataset.iloc[:,:-1].values
    y_train = dataset.iloc[:,1].values
    X_test = dataset1.iloc[:,:-1].values
    y_test = dataset1.iloc[:,1].values
    from sklearn.linear_model import LinearRegression
    regressor=LinearRegression()
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    plt.scatter(X_train,y_train,color="red")
    plt.plot(X_train,regressor.predict(X_train),color="blue")
    plt.title('Linear Regression(training Set)')
    plt.xlabel('X')
    plt.ylabel('Y')
    print(round(sm.r2_score(y_test, y_pred), 2))
    return regressor.predict([[x_new]])
def upload():
    global filename
    global value
    global frame
    global faces, label
    filename = askopenfilename(initialdir = "images")
    pathlabel.config(text="Uploaded image is : " + filename + "\n GET READY FOR THE MUSIC!!!!")
    text.delete('1.0', END)
    orig_frame = cv2.imread(filename)
    orig_frame = cv2.resize(orig_frame, (48, 48))
    frame = cv2.imread(filename,0)
    faces = face_detection.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    #text.insert(END,"Total number of faces detected : "+str(len(faces)))
    if len(faces) > 0:
       faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
       (fX, fY, fW, fH) = faces
       roi = frame[fY:fY + fH, fX:fX + fW]
       roi = cv2.resize(roi, (48, 48))
       roi = roi.astype("float") / 255.0
       roi = img_to_array(roi)
       roi = np.expand_dims(roi, axis=0)
       preds = emotion_classifier.predict(roi)[0]
       emotion_probability = np.max(preds)
       label = EMOTIONS[preds.argmax()]
       messagebox.showinfo("Emotion Prediction Screen","Emotion Detected As : "+label)
    else:
       messagebox.showinfo("Emotion Prediction Screen","No face detceted in uploaded image")

    playsound(str(int(song_prediction("surprise")[0]) + 1) + ".mp3")
    

font = ('times', 20, 'bold')
title = Label(main, text='PERSONALISED MUSIC RECOMMENDATION SYSTEM')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=50)       
title.place(x=0,y=1)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Image With Face", command=upload)
upload.place(x=150,y=200)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=300)


font1 = ('times', 12, 'bold')
text=Text(main,height=10,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()