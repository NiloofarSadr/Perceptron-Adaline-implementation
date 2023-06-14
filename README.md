# Perceptron-Adaline-implementation
Perceptron and Adaptive Linear Neuron (Adaline) Implementation from Scratch
This project is inspired by section two of the book: “Fundamentals of neural networks” (by Laurene V. Fausett ) and aims to implement the basics of neural networks (NN) from the beginning, without using any libraries for creating built-in NN models. The project focuses on the implementation of Perceptron and Adaline from scratch, which is mainly about understanding the basement of a neural network and understanding the first models of feed-forward neural networks and how these models work in detail. This approach can be an effective way for a researcher to understand basic models with all the details.
Model and Dataset: The major goal of the model is to learn the pattern of a set of main characters {A, B, C,…, K} from a set of txt files. Each txt file is a simplified version of an image of a main character that consists of other representative characters. In the training dataset, representative characters are {.,#}). An instance of a train set could be like this:
######. .#....# .#....# .#....# .#####. .#....# .#....# .#....# ######.
(If the above characters do not seem rational to you, copy on a txt file and split each seven characters with \n-enter or see the ReadMe on the code tab)
Which represents character “B”. For the test set, we have the same but with some noises, thus, each main character is represented by {.,#,@, O}. The {O, @} in this set represents the noises. Generally, each data is a text file consisting of 7*9 characters (with mentioned characters for train and test set) that represent a main character (from A to K.) The final goal of the model is to classify main characters by learning from representative characters.
At first, we read all characters from the file and add them to a list. then we replace all points with -1 and all "#" with 1.

1- Perceptron:
we run the Perceptron algorithm on these situations and then discussed the result:
a) Initial values of weights/bias':
In a case bias = 0, As we increased or decreased the initial values of the weights from zero, the accuracy decreased. In contrast, in a case weight = 0, As we increased the initial values of the bias' from zero to 0.13, the accuracy increased and the number of iterations reduced.
b) Threshold value:
we reached the highest accuracy with the least iteration using theta = 0. As we increased or decreased the value of theta, the accuracy decreased.
c) Learning Rate: we reached the highest accuracy via alpha = 0.1.

2- Adaline:
The network is trained using Adalin's rule (Delta). Due to the high learning rate (>0.1), the algorithm did not converge after 2000 iterations.

3- Noise Data:
To avoid the situation where more than one category is selected for the test pattern, for each input file, Y should be in the form of a list of 7 items with just one value of 1 and other values of -1. For this purpose, we initialize Y with -1 and select the category that has the highest value of Y_in, and set the pairwise value in Y to 1. In this case, we are sure that only one category is selected. we reached 0% error on test data using alpha = 0.01, initial weights = -0.02 and bias' = 0.

4- Adaline - feature extraction:
In the previous sections of this exercise, the total number of pixels of each character was used as input to the network. In this section, the projection method is used for feature extraction, and the feature value (instead of pixel values) is given as input to the network. In this method, for each row (and each column) of each character, the sum of pixels On (with a value of one) that row (or column) is counted and the sum value is considered a feature. According to the dimensions of the characters which are 7x9, the number of features of each character will be 16 + 9.
