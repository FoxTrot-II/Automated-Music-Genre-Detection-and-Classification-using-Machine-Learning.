#INTRODUCTION: 

There are basically 10 types of music genres in our day to day life which we use to listen everyday.They are as 
- Metal
- Rock
- Jazz
- Classical
- Pop
- Hip-Hop
- Blue
- Country
- Reggae
- Disco

and in this project the classification of different genres of different music files using python programming language and its tools has been proposed.In this project mainly two python libraries are used named as I-PYTHON and LIBROSA. 


DATASET:

In this project , GT ZAN  data set is used. It comprises of 120 tracks each of 30 seconds in length and the tracks are of 16 bit mono-audio file saved in .wav format( among .wav, .wma and .mp3).Our data-set consist of 1000 files and it is divided in the ratio of 8:2 as 800 files will go under training set and 200 files will go under test set.This data-set is available in  
Kaggle: https://www.kaggle.com/carlthome/gtzan-genre-collection .


LIBRARIES:

1)I-PYTHON   : This library will let us play .wav audio file directly on Jupitor notebook.

2)LIBROSA    : This python library help us to analyse audio signals. It includes nuts and bolts to                             build MIR system. It is used for feature extraction.

3)NUMPY     : It helps to work with the numerical data. This module offers a powerful object called array.

4)KERAS      : It specifically allows you to define multiple input or output models as well as models that share layers. It follows best practices for reducing cognitive load.

5)CSV        : It allows user to import data from a spreadsheet, or another database or data-set in any format.

6)PANDAS    : Pandas is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and                           operations for manipulating numerical tables and time series.

7)OS         : It allows us to use the operating system dependent functionalities and to interact with the underlying operating system in several different ways.
 
8) MATPLOTLIB : It is a plotting library for the Python programming language and its numerical mathematics extension NumPy.


ANALYSATION:

Now our step is to analyse audio files so that we can use it in feature extraction method because feature extraction method can not be done directly on audio files. We will use LIBROSA library for it. So we will analyse mainly two graphs using librosa library i.e

1)WAVEFORM   :  We can plot audio array using librosa.display.waveplot . We will get the amplitude envelope of waveform.

2) SPECTROGRAM :  It is a representation of spectrum of frequencies of sound with time and we will get it by librosa.display.specshow .

After getting these two dependencies we will save these both array to .wav file by librosa.output.writewav so that we can easily access it.


FEATURE EXTRACTION: 

Feature extraction is a process of extensional reduction by which an initial set of raw data is reduced to more manageable groups for processing. In machine learning, pattern recognition and image processing, feature extraction starts from an initial set of measured data and builds derived values intended to be informative and non-redundant. So we will extract features that are relevant to the problem we are trying to solve through the output which we got from the analysation part of audio signals. 
The features are: 

1) Zero Crossing Rate : It is the rate at which audio signals are changing its sign from positive to negative in graph. It usually has higher value for high sound music like Metal and Rock and has lower value for low sound music like Classical and Disco.

2) Spectral Centroid : It indicates where the center of mass of the audio signals is located and is calculated as weighted mean of frequencies. for example C.O.M of BLUES lies in center of the graph whereas the C.O.M of METAL lies at the top of the graph.

3) Spectral Roll-Off : It measures the shape of the signals. It represent the frequency below the specified percent of total spectral energy. 

4) Mel-Freq-Cepstral Coefficient ( MFCC) : These are the small set of features ( i.e 10-20 features in a set ) which describes the overall shape of the spectral envelope

5) Chroma Frequencies : In this all the entire spectral is projected into 12 bins representing 12 distinct semi tones or chromas.

Now before training the model all these features are stored in a .wav file. 

Now we have got the features of audio signals and have to use classification algorithm on the features to classify the songs into different genres.
We can  use either Spectrogramic images or can apply classification model on the extracted features. We will use CNN ( on spectrogramic images ) as it gives better accuracy and consumes less time.  




CONVOLUTION NEURAL NETWORK : 



The whole CNN algorithm is divided into two parts i.e 
1)Convolution 
2)Pooling 
First we will perform convolution process.
We will take the spectrogramic images from the extracted features and extract pixel matrix ( range is between 0 to 155 of pixels) and kernal matrix ( grid of weights) form it . Now we will do the dot product of the pixel matrix and kernal matrix. We overlay kernal matrix over the pixel matrix and get the result as a matrix, same as the size as that of pixel matrix.

Now we will perform the Pooling method.
Again Pooling method is of two types .First one is Max-Pooling and the other one is Min-Pooling. Here we will perform Max Pooling because we have to reduce the size of the inserted array from max to min.The activation function which we used here is RELU activation function. Basically pooling method is done for feature learning and training the model. 

RELU : The rectified linear activation function or ReLU for short is a piece wise linear function that will output the input directly if it is positive, otherwise, it will output zero.

On pooling method , we will give the super imposed matrix ( I.e the matrix after getting by doing the dot product of kernal matrix and pixel matrix ) as the input and gets the output as the optimized matrix which is smaller in size than the input matrix. We will keep doing this process until we will get the maximum optimized matrix.
After performing the max pooling we will get the fully connected layers of features of different genres.

The CNN is divided into 3 layers : 
1)INPUT LAYER : We used 512 nodes in input layer with RELU as the activation function.
2)HIDDEN LAYER: All computational process occurs in this layer. Again RELU is used and divided          into 3 layers each of 256 , 128 and 64 nodes.
3)OUTPUT LAYER: This contains 10 nodes. We use SOFTMAX activation function . Optimizer used here is ADAM and SPARSE CATEGORIAL CROSS-ENTROPY as the loss function.

Cross-Entropy: If your Yi’s are not encoded , use categorical cross-entropy ( for a 3 class ) 
[1 0 0]  [0 1 0]  [0 0 1]
But if your Yi’s are integer , we use sparse categorical cross-entropy.
[ 1 ]   [ 2 ]   [ 3 ]
It depends on how you load your dataset. One advantage of using sparse.c.c is that it saves time in memory as well as computational because it simply use single integer for class rather than whole integer.




Now we have trained our model with the trained set and got 82.25 % accuracy . Then we compare it with the test set which gave the accuracy of 59.5%. Since the test accuracy is much much lower than the trained accuracy , this shows that the over-fitting has occurred. To remove this over-fitting we did the validation approach by dividing the trained set into train set and validation set in the ratio of 1:3 .
Then we build 5 layer CNN model to train it with the validation set . The optimizer used here is ADAM optimizer . Sparse Cross-Entropy is used as loss function . The accuracy here after validation approach is 64.5% which is more than the earlier we got.

