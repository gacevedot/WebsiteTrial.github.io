# WebsiteTrial.github.io
This is a Neural Network for classifying dysarthric vs. healthy speech recordings. The model can be trained using features based on the fundamental frequency, the logarithmic Mel Spectrum, or the Bark Band Spectrogram

## Motivation
Dysarthria is a speech impairment that is caused by improper muscle functioning, which causes muscles involved during speech production to be imapired. The improper functioning of these muscles will casue multiple symptoms that can be reflected on the subject's speech. Descriptors extracted from audio recordings (waveform or spectrogram) can be used as a source of information related to the muscle functioning and the speech capacity of the person. It is theorized that the mechanisms of impairment due to dysarthria can be detected by acoustic measures, such as the features used in this project. Nowadays, there is a great interest in understanding what features can better describe the impairments resulting from dysarthria. These features can be used to modeldysarthric speech, which can then be used for detecting its presence in speech recordings. 
## Objective 
The main objective of these project is to use acoustic features from subjects with stroke-related dysarthria symptoms and healthy subjects to train a CNN model for detecting the presence of dysarthria from speech recordings. Different type of features were used to compare the performance of the models, and to understand what features are better to detect dysarthria in stroke patients. 
## Dataset 
The dataset consisted of 1080 audio recordings of the sentences "Buy Bobby a puppy" from a different healthy subjects and stroke survivors. 
This file contains the functions to load the audio files (.wav) and perform data augmentation for all the audio recordings. This was necessary due to the small amount of data available and to get more robusts results 
### Injecting Noise 
A Normal Gaussian Noise was added to all the recordings to create a new augmented sample. A random signal was created using numpy and then added to the audio signal
### Lowpass Filter
The idea of filtering all the audio recordings was to create muffled samples. These was done using a sixth order lowpass bessel filter with a cutoff frequency a 4000Hz

After data augmentation, a total of 1080 samples was used for the final dataset 

## Feature Extraction
The features were extracted using the [surfboard](https://github.com/novoic/surfboard) library. The extracting_feature file contains the functions to extract all the features from the .wav files
### f0_contour_static
This functions extracts features related to the fundamental frequency: mean, std, max, skewness, kurtosis, first_derivative_mean, first_derivative_std, The fundamental frequency is related ton the vocal fold vibrations during speech production, which are controlled by the laryngeal musculature. The fundamental frequency is closly related to the pitch in the sound produced. Thus, changes on features associated with the fundamental frequency can provide information about the function of this muscle group and pitch production capacity of the subject. 
The output of the function is a (7,) numpy array. 
### logmel_spectrogram_static
The features related to the Mel Spectrogram: mean, std, first_derivative_mean, first_derivative_std
The Mel Spectrogram is a spectrogram that represents the frequencies in the Mel scale. Humans perceive sound frequencies in a non-linear manner. In other words, we are not able to here the difference between a pair of frequencies (e.g. 10,000 Hz and 10,000 Hz), like we do of another pair of frequencies with equal distance between them (e.g. 500 Hz and 1,000 Hz). The Mel Scale is a proposed unit of pitch such that it represents the pitch how it is heard by the listeners. 
During speech production, many muscles are involved to produce movement of the articulators (jaw, tongue, lips, palatale, etc.). The position of these structures and the velocity will determine the sound characteristics and the transitions between sounds. The frequencies captured aside from the f0, and their energy distributions will provide information about the position of this articulators and the movement capacity of the subjects. 
### bark_spectrogram_statics
The features related to the Bark Spectrogram: mean, std, first_derivative_mean, first_derivative_std
The Bark Spectrogram is related to the Mel Spectrogram. The same idea follows, however the Bark scale is divided in 24 critical bands. 
Just like the Mel Spectrogram, there is articulatory information in the Bark Band Spectrogram. This can provide information about the speech production capacity of the person.

## Convolutional Neural Network Components 
For this project, a one-dimension Convolutional Neural Network model was created to improve the recognition of dysarthric speech using acoustic features. The inputs fed to the model consisted of a R^D vector, the dimensions depended on the group of features selected. The python framework was written using the kersas library for Deep Learning. 
### Layers
1. _Convolutional Layers_: The input layer and three of the hidden layers are 1D Convolutional Layers. These layers were chosen because the features related to the scaled Spectrograms are a series of frequencies that lay next to each other, forming the new scale. These layers extract information about the input features by convolving a filter with a kernel size of 5 over on spatial dimension. The filter is shifted across the input features by 1 stride, and then the dot product is calculated. The output of these layers are feature maps which wil become inputs a successive pooling layer. 
2. _Pooling Layers_: Four of the hidden layers are 1D Maximum Pooling layers. These layers are applied to downsample output feature maps from the previous convolutional layers. The max pooling consists on taking the maximum value over a window of size 5, which is shifted to the left by 2 strides. The output of the max pooling layers are downsampled feature maps that are sensitive to the most present feature in the window. 
3. _Flatten Layer_: The flatten layer is placed after the last max pooling layer and before the densly connected network. This layer will flatten the input (max pooling layer output) converting the data into a one-dimensional array, without affecting the batch size. The output will consist of a single long feature vector, which will be the input to the fully-connected layer. 
4. _Dense layers_: These layers are fully connected layers and represent one hidden and the output layers. Each neuron of the layer is densely connected to the previous layer. In other words, each of the neurons present in a dense layer will receive an input from all the neurons in the previous layer. These connections allow the dense layers to provide feature learning from all possible combinations of the features coming from previous layers. The dense layers compute the dot products between the input and the weight maztrix, which is learned during the training process. 
5. _Dropout layers_: The two drop pout layers work as regularization layers to prevent overfitting, which is a major concerns in this project. These layers will randomly ignore some number of the layer outputs (input fractions of 0.3, and 0.2). The layer will randomly select inputs and set them to 0. This process causes the training process to be "noisy". 
### Activation Functions 
The activation functions of the layer will define the output given an input:
1. _ReLu_: The "ReLu" (Rectified Linear Unit) activation function was used for the Convolutional layers and the dense hidden layer. This function is classified as a ridge activation, which are multivariate functions that act on linear combinations of the input features. This functions is placed to introduce non-linearity to the network, and will replace the negative values in the feature map with zeros (max(x,0)). 
2. _Sigmoid_: The Sigmoid, or Logistic, activation function was applied to the output dense layer to produce a probability output. This functions is used for a nonlinear activation, and transform the input into a value between 0.0 and 1.0. The activation function properties provides a two-class logistic regression for the output layers.
### Loss Function
A _logarithmic loss function_ (binary_crossentropy, in Keras) is used for the training process. This loss function was selected because it assess the performance of the model by taking the log of probability values between 0 and 1, which fits with our final output. The properties of the log will make the cross-entropy loss increase as the predicted probability diverges from the true label. 
### Optimizer 
The _Adam_ optimizer was used for the model. These optimization algorithm is a adaptive learning rate version of the SGD with momentum. It uses the squared gradients to scale the learning rate and the gradients moving average as momentum. 
### Reduce Learning Rate On Plateau
The function _ReduceLROnPlateau()_ was added to the model for learning rate scaling. This callback will monitor the validation loss throughout the training, and if there is no improvement (the validation loss is "on plateau"), it will reduce the learning rate by a factor of 0.4. This callback will avoid the learning process to become stagnant. 
### Kernel Regularization
An _L2 Norm_ regularization was applied to the first fully-connected layer (dense layer after flattening). The addition of this regularizer was added to prevent overfitting, which is a major concern for this project. This regularization technique applies L2 regularization penalty on the layer's kernel. 
### Feature Learning 
The input layer and the successive 8 layers (composed of convolutional layers and maximum pooling layers) form the __feature learning__ of the CNN. The combination of convolutional layers followed by pooling layers allow the model to learn in a supervised manner the representations needed for classification. During training, this process enables the model to understand the input features and provide the best representations that will be fed to the classification process.
### Classification
The classification portion of the model is carried from the Flattening of the feature learning final output and the fully connected layers. This portion of the model learns non-linear combinations of the high-level representations from the convolutional layers' output. On each iteration during training, the flattened array serves as the input for a combination of a feed-foward NN and backpropagation. This process helps the model to learn the weights of the features for the classification process. In other words, the values that will be used for the dot product with the input to provide a label for classification. 
The final model for this project should work as a binary classifier. The target labels are 0 or 1, 0 being healthy speech and 1 dysarthric speech. Thus, the output of the _output layer_  should provide a label 
