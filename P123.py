#Importing
import cv2
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#Pil = Pillow Package
from PIL import Image
import PIL.ImageOps
import os, ssl, time

#Fetching the data
X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)

#Splitting the data into training and testing
Xtrain, Xtest, yTrain, yTest = train_test_split(X, y, random_state=9, train_size=3500, test_size=500)
#Scaling the features
XtrainScaled = Xtrain/255.0
XtestScaled = Xtest/255.0

#Fitting the training data into the model
clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(XtrainScaled, yTest)

#Making a prediction and printing the accuracy
yPred = clf.predict(XtestScaled)
accuracy = accuracy_score(yTest, yPred)
print("The accuracy is: ",accuracy)

#Starting the camera and reading the frames
cap = cv2.VideoCapture(0)
while(True):
    try:
        ret, frame = cap.read()
        
        #Drawing a box in the centre of the screen and considering the area only inside the box to detect images
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upperLeft = (int(width/2 - 56), int(height/2 - 56))
        bottomRight = (int(width / 2 + 56), int(height / 2 + 56))
        cv2.rectangle(gray, upperLeft, bottomRight, (0, 255, 0), 2)
        roi = gray[upperLeft[1] : bottomRight[1], upperLeft[0]:bottomRight[0]]  
        
        #Converting the cv2 image to PIL format
        imPIL = Image.fromarray(roi)  
        
        #Converting to grayscale image format
        imagebw = imPIL.convert('L')
        imagebwResized = imagebw.resize((28,28), Image.ANTIALIAS)
        imagebwResizedInverted = PIL.ImageOps.invert(imagebwResized)
        pixelFilter = 20
        min_pixel = np.percentile(imagebwResizedInverted, pixelFilter)
        image_bw_resized_inverted_scaled = np.clip(imagebwResizedInverted-min_pixel, 0, 255)
        max_pixel = np.max(imagebwResizedInverted)
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        testPred = clf.predict(test_sample) 
        print("The Predicted class is: ", testPred) 
        
        #Displaying the resulting frame
        cv2.imshow('frame', gray)
        #displays for a millisecond
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass
    
#When everything is done, releasing the capture
cap.release()    
cv2.destroyAllWindows()                                