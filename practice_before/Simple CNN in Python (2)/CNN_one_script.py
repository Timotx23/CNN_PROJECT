# Import necessary libraries
import torch 
import matplotlib.pyplot as plt


# Function 1: Convolution
def apply_convolution(input_data, filters, stride=1):
    """
    Apply convolution on a 3D input using given filters.

    Parameters:
    - input_data: 2D numpy array, the input data (channels, length)
    - filters: List of 2D numpy arrays, representing the filters
    - stride: Integer, the stride of the convolution operation

    Returns:
    - output: 2D numpy array, the result of convolution
    """

    # Ensure filters are the right shape
    ## YOUR CODE HERE (Hint: assert might be useful here)

    # Initialize an empty array to store the convolution result
    input_len = input_data.shape[1]
    filter_len = filters[0].shape[1]

    output_len = (input_len - filter_len) // stride + 1
 ## YOUR CODE HERE # integer division is like floor operation
    num_filters = len(filters) ## YOUR CODE HERE
    output = torch.zeros((num_filters, output_len))## YOUR CODE HERE 

    # Perform convolution with the specified stride
    for i in range(output_len): # Loop over output length
        start = i * stride #starting point
        end = start + filters[0].shape[1] #end point -> The window width must match the filter width.

        window = input_data[:, start:end] #the window which acts like a limiter -> it allows the system to just view one part of the system

        for f, filter_ in enumerate(filters): # this loops over the filters again
            output[f, i] = torch.sum(window * filter_) #adds these enumerated filters to the output -> this output is of the type array
        ## YOUR CODE HERE 
    return output

# Function 2: 1D Pooling
def apply_pooling(input_data, pool_size, stride=1):
    """
    Apply 1D pooling on the input data.

    Parameters:
    - input_data: 2D numpy array, the input data
    - pool_size: Integer, size of the pooling window
    - stride: Integer, the stride of the pooling operation

    Returns:
    - result: 2D numpy array, the result after pooling
    """
    # Ensure the input is a 1D or 2D array
    ## YOUR CODE HERE (remember to check ndim)

    # Calculate the output length after pooling
    input_size = input_data.shape[1]
    
    output_length = (input_size - pool_size) // stride + 1## YOUR CODE HERE

    # Initialize an empty array to store the pooling result
    result = torch.zeros((input_data.shape[0], output_length))## YOUR CODE HERE

    # Perform 1D pooling with the specified stride
    for i in range(output_length):
        start = i * stride #declare the start of the pool
        end = start + pool_size # declare the end of the pool

        window = input_data[:, start:end] # declare the window based on which the ouput will look at

        result[:, i] = torch.max(window, dim=1).values# For each channel, it takes the maximum value over the pooling window and stores those values in the i-th position of the output tensor.
        #This needs dim because it needs to find the max value of each channel
    return result


# Function 3: ReLU Activation
def apply_relu(input_data):
    """
    Apply ReLU activation on the input data.

    Parameters:
    - input_data: 2D numpy array, the input data

    Returns:
    - result: 2D numpy array, the result after applying ReLU
    """
    # Apply ReLU activation element-wise 

    return torch.relu(input_data) 
    #torch.relu -> doesnt need to be prestated ie it can be called directly
    #torch.nn.ReLU needs to be declared before ie relu = torch.nn.ReLU() and then do relu(input_data)