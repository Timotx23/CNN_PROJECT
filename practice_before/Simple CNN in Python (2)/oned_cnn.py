import torch
from CNN_one_script import apply_convolution, apply_pooling, apply_relu
# Function 4: 1D CNN

def one_dimensional_cnn(input_data, filters_layer1, filters_layer2, pool_size, conv_stride=1, pool_stride=1):
    """
    Create a 1D CNN with two convolutional layers, ReLU activation, and pooling.

    Parameters:
    - input_data: 2D numpy array, the input data
    - filters_layer1: List of 2D numpy arrays, filters for the first convolutional layer
    - filters_layer2: List of 2D numpy arrays, filters for the second convolutional layer
    - pool_size: Integer, size of the pooling window

    Returns:
    - output: 2D numpy array, the final output of the CNN
    """
    # Apply convolution and ReLU activation for the first layer
    layer1_output = apply_convolution(input_data, filters_layer1, conv_stride)
    layer1_output_relu = apply_relu(layer1_output)
    layer1_output_pooled = apply_pooling(layer1_output_relu, pool_size, pool_stride)

    # Apply convolution and ReLU activation for the second layer
    layer2_output = apply_convolution(layer1_output_pooled, filters_layer2, conv_stride)
    layer2_output_relu = apply_relu(layer2_output)
    layer2_output_pooled = apply_pooling(layer2_output_relu, pool_size, pool_stride)

    torch.set_printoptions(precision=2, suppress=True)
    print(f"Input data: {input_data}\n")
    print(f"Output convolution 1:\n{layer1_output}")
    print(f"ReLU 1:\n{layer1_output_relu}")
    print(f"Pooling 1:\n{layer1_output_pooled}\n")
    print(f"Output convolution 2:\n{layer2_output}")
    print(f"ReLU 2:\n{layer2_output_relu}")
    print(f"Pooling 2:\n{layer2_output_pooled}\n")

    return layer2_output_pooled