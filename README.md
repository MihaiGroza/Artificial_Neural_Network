# Artificial_Neural_Network

## Goal

Apply an Artificial Neural Network model on Wisconsin Breast Cancer Data to predict if tumour is Benign or Malignant. 

Data set consists of 699 instances with 9 relevant attributes.
Training and testing split is 80-20

## Neural Network Structure

The Neural Network will contain 3 layers

1 input layer with a rectifier as activation function (9 nodes)
 
1 hidden layer with a rectifier as activation function (5 nodes)

1 output layer with a sigmoid as activation function   (1 node for binary prediction)

### Visual Representation of model structure

![Image of model](https://github.com/MihaiGroza/Artificial_Neural_Network/blob/master/model_plot.png)


## Results

The initial Testing Accuracy of the model was 0.963

However, I used 10 fold CrossValidation on the model for a more reliable testing accuracy.

It resulted in this set of accuracies:
[0.98181818, 0.96363637, 0.96363635, 0.92727273, 0.94545454,0.94444444, 0.96296295, 1, 0.94444443, 0.98148148]

with a mean of 0.9615, which is close to the initial testing accuracy.

