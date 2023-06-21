# XOR-Neural-Network
## About
I implemented a neural network with one hidden layer and one output layer from scratch (NumPy) in order to classify 4 points in order to solve the
XOR problem by classifying the points (1,1) and (0,0) into one class, and points (0,1), (1,0) into another class. 

## Implementation
The weights for our model were initialized randomly with values between 0 and 1. The backpropagation algorithm was coded in NumPy, as was the model architecture. Training loss was brought down to around 0.07, with 1000 epochs and a learning rate of 1. When ran repeatedly the model consistently reached 100% classification accuracy. I found that these respective hyperparameter values yielded the best performance during inference. The decision boundaries classifying the points into their respective classes were constructed through the use of a mesh grid that used a red-blue color scheme for the boundary separating the points of each class. For fun, I decided to introduce noise with a standard deviation of 0.5, 1, and 2 into the data to test my algorithm on a more "real-life" dataset, but could not produce optimal results. At best, when noise of standard deviation of 1 was introduced into the training data, the model was able to classify the points with a 69% accuracy. 

## Frameworks/Tools
• NumPy <br>
• Jupyter Notebook <br>
• Matplotlib <br>
