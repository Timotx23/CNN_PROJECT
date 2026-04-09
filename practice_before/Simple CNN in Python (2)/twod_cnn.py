import torch
from CNN_one_script import apply_convolution, apply_pooling, apply_relu
def apply_2D_convolution(input_data, filters, stride=1):
    """
    Apply convolution on a 3D input (2D image + channels) using given filters.

    Parameters:
    - input_data: 3D numpy array, the input data (channels, height, width)
    - filters: List of 3D numpy arrays, representing the filters
    - stride: Integer, the stride of the convolution operation

    Returns:
    - output: 3D numpy array, the result of convolution
    """

    # If input_data is 3D, ensure filters are the right shape
    ## YOUR CODE HERE (Hint: assert might be useful here)
    assert len(filters) > 0, "Need at least one filter"

    # Initialize an empty array to store the convolution result
    input_len_1 =  input_data.shape[1]
    input_len_2 =  input_data.shape[2]
    filter_len_1 = filters[0].shape[1]
    filter_len_2 = filters[0].shape[2]
    
    output_h = (input_len_1- filter_len_1)// stride +1
    output_w = (input_len_2 - filter_len_2)//stride + 1 
    num_filters = len(filters)
    output = torch.zeros((num_filters, output_h, output_w))

    # Perform convolution with the specified stride      
   
    for i in range(output_h):
        for j in range(output_w):
            start_h= i *stride
            start_w = i * stride
            end_h= start_h + filter_len_1
            end_w = start_w + filter_len_2
            window = input_data[:, start_h:end_h, start_w: end_w]
            for f, filterss in enumerate(filters):
                output[f,i,j] = torch.sum( window* filterss)


    return output

#2d pooling
def apply_2D_pooling(input_data, pool_size, stride=1):
    """
    Apply 2D pooling on the input data.

    Parameters:
    - input_data: 3D numpy array, the input data (channels, height, width)
    - pool_size: Integer, size of the pooling window
    - stride: Integer, the stride of the pooling operation

    Returns:
    - result: 3D numpy array, the result after pooling
    """
    # Ensure the input is a 3D array
    ## YOUR CODE HERE

    # Calculate the output dimensions after pooling
    input_size_1 = input_data.shape[1]
    input_size_2 = input_data.shape[2]
   
    output_h = (input_size_1 - pool_size) // stride +1 ## YOUR CODE HERE
    output_w = (input_size_2 - pool_size) // stride + 1## YOUR CODE HERE

    # Initialize an empty array to store the pooling result
    result = torch.zeros((input_data.shape[0], output_h, output_w))## YOUR CODE HERE

    # Perform 2D pooling with the specified stride
    for i in range(output_h):
        for j in range(output_w):
            start_h= i *stride
            start_w = j * stride
            end_h= start_h + pool_size
            end_w = start_w + pool_size
            window = input_data[:, start_h:end_h, start_w: end_w]
            result[:, i, j] = torch.amax(window, dim=(1,2)) ## YOUR CODE HERE
    return result


def two_dimensional_cnn(input_data, filters_layer1, filters_layer2, pool_size, conv_stride=1, pool_stride=1):
    """
    Create a 2D CNN with two convolutional layers, ReLU activation, and pooling.

    Parameters:
    - input_data: 3D numpy array, the input data (channels, height, width)
    - filters_layer1: List of 3D numpy arrays, filters for the first convolutional layer
    - filters_layer2: List of 3D numpy arrays, filters for the second convolutional layer
    - pool_size: Integer, size of the pooling window

    Returns:
    - output: 3D numpy array, the final output of the CNN
    """
    # Apply convolution and ReLU activation for the first layer
    layer1_output =  apply_2D_convolution(input_data, filters_layer1, conv_stride)
    layer1_output_relu = apply_relu(layer1_output)
    layer1_output_pooled = apply_2D_pooling(layer1_output_relu, pool_size, pool_stride)

    # Apply convolution and ReLU activation for the second layer
    layer2_output = apply_2D_convolution(layer1_output_pooled, filters_layer2, conv_stride)
    layer2_output_relu = apply_relu(layer2_output)
    layer2_output_pooled = apply_2D_pooling(layer2_output_relu, pool_size, pool_stride)

    torch.set_printoptions(precision=2,  sci_mode=True)
    print(f"Input data shape: {input_data.shape}\n")
    print(f"Output convolution 1 shape: {layer1_output.shape}")
    print(f"Pooling 1 shape: {layer1_output_pooled.shape}\n")
    print(f"Output convolution 2 shape: {layer2_output.shape}")
    print(f"Pooling 2 shape: {layer2_output_pooled.shape}\n")

    return layer2_output_pooled