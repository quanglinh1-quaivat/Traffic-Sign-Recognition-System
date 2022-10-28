# Traffic-Sign-Recognition-System
2. Method and data 

2.1 DATA
Overview:
Single-image, multi-class classification problem
More than 40 classes
More than 50,000 images in total
Large, lifelike database
Reliable ground-truth data due to semi-automatic annotation
Physical traffic sign instances are unique within the dataset
The training set archive is structures as follows:
One directory per class
Each directory contains one CSV file with annotations ("GT-<ClassID>.csv") and the training images
Training images are grouped by tracks
Each track contains 30 images of one single physical traffic sign
Image format
The images contain one traffic sign each
Images contain a border of 10 % around the actual traffic sign (at least 5 pixels) to allow for edge-based approaches
Images are stored in PPM format (Portable Pixmap, P6)
Image sizes vary between 15x15 to 250x250 pixels
Images are not necessarily squared
The actual traffic sign is not necessarily centered within the image.This is true for images that were close to the image border in the full camera image
The bounding box of the traffic sign is part of the annotations (see below)
Annotation format
Annotations are provided in CSV files. Fields are separated by ";"   (semicolon). Annotations contain the following information:
Filename: Filename of corresponding image
Width: Width of the image
Height: Height of the image
ROI.x1: X-coordinate of top-left corner of traffic sign bounding box
ROI.y1: Y-coordinate of top-left corner of traffic sign bounding box
ROI.x2: X-coordinate of bottom-right corner of traffic sign bounding box
ROI.y2: Y-coordinate of bottom-right corner of traffic sign bounding box
The training data annotations will additionally contain
ClassId: Assigned class label

2.2 Methods 
Before preprocessing the training dataset was equalized making examples in the classes equal as it is shown on the figure below. Histogram of 43 classes for training dataset with their number of examples for Traffic Signs Classification before and after equalization by adding transformated images (brightness and rotation) from original dataset. After equalization, the training dataset has increased up to 86989 examples.
 

Convolution only takes one input with a fixed channel instead of all channels so it only learns a small part of the input channel, reducing performance. So we need to shuffle data then Convolution can learn more about other channels.
Example: 
 
Histogram equalization is a basic image processing technique that adjusts the global contrast of an image by updating the image histogram’s pixel intensity distribution. Doing so enables areas of low contrast to obtain higher contrast in the output image.
 
Then we calculate mean and STD and normalize by dividing by 255.
Resulted preprocessed nine files are as follows:
- data0.pickle - Shuffling.
- data1.pickle - Shuffling, /255.0 Normalization.
- data2.pickle - Shuffling, /255.0 + Mean Normalization.
- data3.pickle - Shuffling, /255.0 + Mean + STD Normalization.
- data4.pickle - Grayscale, Shuffling.
- data5.pickle - Grayscale, Shuffling, Local Histogram Equalization.
- data6.pickle - Grayscale, Shuffling, Local Histogram Equalization, /255.0 Normalization.
- data7.pickle - Grayscale, Shuffling, Local Histogram Equalization, /255.0 + Mean Normalization.
- data8.pickle - Grayscale, Shuffling, Local Histogram Equalization, /255.0 + Mean + STD Normalization.
Datasets data0 - data3 have RGB images and datasets data4 - data8 have Gray images.
Shapes of data0 - data3 are as following (RGB):
- xtrain: (86989, 3, 32, 32)
- ytrain: (86989,)
 
- xvalidation: (4410, 3, 32, 32)
- yvalidation: (4410,)
 
- xtest: (12630, 3, 32, 32)
- ytest: (12630,)
 
Shapes of data4 - data8 are as following (Gray):
- xtrain: (86989, 1, 32, 32)
- ytrain: (86989,)
 
- xvalidation: (4410, 1, 32, 32)
- yvalidation: (4410,)
 
- xtest: (12630, 1, 32, 32)
- ytest: (12630,)
We separated the data to training, validation, testing set like the picture below: 

Mean image and standard deviation were calculated from the training dataset and applied to validation and testing dataset for appropriate datasets. When using image for classification, it has to be preprocessed firstly in the same way and in the same order according to the chosen dataset among nine.
(Data: https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-preprocessed)

3. System Design & Implementation 

3.1 Backend:
		The main idea of our project used CNN(Convolutional Neural Network) to classify traffic signs. Keras is a Python package that is a wrapper for deep learning libraries such as TensorFlow and Adam, … for model building, training, optimization.
Model: 

We use Keras to build an CNN model have layers: Conv2D, Maxpool2D, BatchNormalization, Flatten,  Dense

Convolutional layer: 
The core block of a CNN architecture is the convolutional layer. It involves a set of kernels (or filters) that can only receive a small fraction but extend through the full depth of input volume. Each filter is learnable and will be involved across the height and width of input volume along with the computation of dot product between the entries of filters and the input. During the initial process, a 2D feature map will be produced. Finally, the network will learn from activated filters while detecting some specific features at some spatial position in the input. The full output of the convolutional layer is formed by storing the activation maps for all filters along the depth dimension. The output of a neural is a small region in the input and the parameters will be shared between neurons in the same activation map. 

Maxpooling2D:
Similarly, the pooling layer is responsible for reducing the spatial size of the convolved feature. It not only decreases the computational consumption through reducing dimensionality, but also is used to extract dominant features. Besides, the pooling layer serves to simplify the configuration of parameters and memory footprint, and thus controls overfitting.

There are two dominant types of pooling, including max pooling. As the name implies, the max pooling returns the maximum value from the portion of the image covered by the kernel. The average pooling returns the average across all the values from the portion of the image covered by the Kernel. The depth of volume is not changed. 

Flatten:
Flattening is used to convert all the resultant 2-Dimensional arrays from pooled feature maps into a single long continuous linear vector. The flattened matrix is fed as input to the fully connected layer to classify the image.

Fully Connected Layer:
In the fully connected layer, the neurons connect to all activations as seen in the regular neural networks. Generally, inserting a fully connected layer is a cheap way to capture the nonlinear combination of high-dimensional features as represented by the yield of CONV.

ReLU:
Rectified linear unit (ReLU) utilized the non-saturating activation function 𝑓(𝑥) = max(0, 𝑥) to remove negative values from an activation map and replace them with number 0. This operation enhances the nonlinearity of the decision function and the whole network without influencing the reception domain of the CONV layer. . ReLU has gained more popularity compared to other functions such as sigmoid, tanh,..  due to the faster speed of training the neural network .

Batch Normalization and Dropout: 
Batch normalization(BN) has been known to improve model performance, mitigate internal covariate shift, and apply a small regularization effect. Such functionalities of the BN and empirical studies proving the effectiveness of BN helped to solidify people's preference of using BN over dropout.
Dropout is meant to block information from certain neurons completely to make sure the neurons do not co-adapt.


About Keras: https://www.tensorflow.org/tutorials/images/cnn

3.2 Front-End: 
Python has a lot of GUI frameworks, but Tkinter is the only framework that’s built into the Python standard library. Tkinter has several strengths. Visual elements are rendered using native operating system elements, so applications built with Tkinter look like they belong on the platform where they’re run. Tkinter is lightweight and relatively painless to use compared to other frameworks. This makes it a compelling choice for building GUI applications in Python, especially for applications where a modern sheen is unnecessary, and the top priority is to quickly build something that’s functional and cross-platform.
Make a simple GUI to classify one image each time with Tkinter package in python:



Tkinter: https://realpython.com/python-gui-tkinter/

4. Results and Discussion 

After training data:
Fitting data of validation and training set:

Each epoch with accuracy, val_accuracy, loss, val_loss when running program:
val_loss is the value of cost function for your cross-validation data
loss is the value of cost function for your training data 
val_accuracy is the accuracy of a batch of testing data
accuracy is the accuracy of a batch of training data 



Figure represent accuracy, val_accuracy, loss, val_loss:


Result: 
Test on testing data: we need to confirm that model runs correctly with testing data when validation data was fit with high accuracy 


Test with GUI, real-time test with camera: We make a simple GUI to test model with other image that collect on the internet to evaluate the accuracy of the model with external data.


Perspective:
The result runs with high accuracy with testing data, quite good with GUI but real-time is worse. Some labels can not be detected correctly. Real-time testing needs a lot of adjusting for accurate results.
We just only explored this problem from a limited researching angle. There are a lot of real-world difficulties while recognizing traffic signs, including illuminant issues, rotations, partial occasions and physical damages, etc. There are still a lot of research spaces in this field to be mined. 

5. Conclusion and Perspectives

In Summary, we think the Convolutional Neural Network (CNN)  is highly suitable for classification of images. CNN effectively uses adjacent pixel information to effectively downsample the image first by convolution and then uses a prediction layer at the end. For example, SVM are margin classifiers and support different kernels to perform these classification. Where SVM suffers is predicting the class labels when in this project, there are many class labels to classify then SVM is not suitable.
Our team has successfully developed a simple traffic sign recognition. Study various image processing methods in geometrical recognition, and the CNN algorithm, which created the Traffic sign recognition program. As in a research project, the system can only function on data testing files with a limited number of traffic signs. Another drawback to the application is that it cannot function on hardwares that does not meet the requirements regarding processing, camera quality and recognition capabilities.
In this project, We have 43 class labels of traffic signs but in real life, there are many traffic signs and many labels. The number of training sets will also be increased, and better quality control will be applied to increase overall accuracy. The system will also be developed to recognize and identify more traffic signs. If we have more time and members, we hope we can improve our algorithms, improve performance, with more knowledge, we hope we can create a program that can convert label name to voice notification which helps drivers to drive easier and avoid missing traffic signs. In the future, the team hopes to enhance traffic sign identification to handle damaged and obstructed signs.
