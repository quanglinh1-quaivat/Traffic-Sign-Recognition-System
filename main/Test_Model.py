#import library
import matplotlib.pyplot as plt
from keras.models import load_model
import pickle
import numpy as np
import pandas as pd

def label_text(file):
    # Defining list for saving label in order from 0 to 42
    label_list = []

    # Reading 'csv' file and getting image's labels
    r = pd.read_csv(file)
    # Going through all names
    for name in r['SignName']:
        # Adding from every row second column with name of the label
        label_list.append(name)

    # Returning resulted list with labels
    return label_list

#read test data % load model
with open(r"C:\Users\LAPTOP88\Downloads\Traffic_Sign_Detection\Org_data\data2.pickle",'rb') as f:
    data = pickle.load(f, encoding='latin1')
data['x_test'] = data['x_test'].transpose(0, 2, 3, 1)
labels = label_text(r"C:\Users\LAPTOP88\Downloads\Traffic_Sign_Detection\Org_data\label_names.csv")
model = load_model("FinalModel(5).h5")

#predict result
pred = model.predict(data['x_test'])

#plot some result
plt.figure(figsize = (25, 25))

start_index = 0
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    prediction = np.argmax(pred[start_index + i])
    col = 'g'
    plt.xlabel('Name={}'.format(labels[prediction]), color = col)
    plt.imshow(data['x_test'][start_index + i])
plt.show()

