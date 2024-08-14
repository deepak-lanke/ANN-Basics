## Understanding the basics of Artificial Neural Networks (ANN)

*Modelling of ANN for Beale function*

#### *What is a Beale function?*

The Beale function is multimodal, with sharp peaks at the corners of the input domain.

![Beale function](https://www.sfu.ca/~ssurjano/beale.png)

- *Dimensions:* 2
- *Input Domain:* The function is usually evaluated on the square $x_i ∈ [-4.5, 4.5]$, for all $i = 1, 2$.
- *Global minimum:* $f(3,0.5)=0$

*For more information [click here](https://en.wikipedia.org/wiki/Test_functions_for_optimization)*

### Data preprocessing

- The input and output data were scaled between 0 – 1. To normalize the data, `MinMaxScaler` tool was imported `sklearn.preprocessing` package. Then the scaled data was stored for further usage.
- The scaled data was split into input data and target data required for the development of the ANN model. 70% of the data was taken as the input data, 15% as validation and the rest 15 % as testing data.

### ANN modelling

ANN modelling was implemented for the scaled data using TensorFlow package in python. Sequential and Dense functions were imported from TensorFlow and it was used to create the neural network.

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
```
The parameter of model such as hidden layers, nodes, activation functions, epochs, sample size for training were varied to see the performance of the model.

### Results

1. Effect of Number of Hidden Layers
2. Effect of Number of Hidden Nodes
3. Effect of Activation Functions
4. Effect of Number of epochs
5. Effect of Sample size for training