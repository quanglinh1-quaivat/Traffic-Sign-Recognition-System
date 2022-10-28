#import library
import cv2
from keras.models import load_model
model = load_model("testing(2).h5")
import numpy as np
frameWidth= 640
frameHeight = 480
brightness = 180
threshold = 0.95
font = cv2.FONT_HERSHEY_SIMPLEX

#define camera
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

#get 43 labels name
labels = open(r"C:\Users\LAPTOP88\Downloads\Traffic_Sign_Detection\Traffic_Sign\label_names.csv").readlines()
labels = labels[1::]
lbl=[]
for label in labels:
    lbl.append(label.split(',')[1].rstrip('\n'))

while True:

# READ IMAGE
    success, imgOrignal = cap.read()
    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32,32))
    expand_input = np.expand_dims(img,axis=0)
    img = np.array(expand_input)
    img = img/255

    # display results
    # cv2.imshow("Processed Image", img)
    cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    predictions = model.predict(img)
    print(predictions.argmax())

    y_classes = [np.argmax(element) for element in predictions]
    probabilityValue = np.amax(predictions)
    if probabilityValue > threshold:
    # print(getCalssName(classIndex))
        cv2.putText(imgOrignal, str(y_classes) + " " + str(lbl[y_classes[0]]), (120, 35), font, 0.75, (0, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Result", imgOrignal)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
