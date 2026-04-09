from skimage import io, color # If required, install scikit-image with pip install -U scikit-image
import matplotlib.pyplot as plt
import twod_cnn
import torch
def visualize_image():
    # first let's load and visualize an image (we'll use a famous img :P)

    img_path = 'Lenna_(test_image).png'

    # load and plot the image
 
    img = io.imread(img_path)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

    # let's print the shape of the image
    print("Image shape:", img.shape)  # Should print (512, 512, 3) since it's a RGB image (1 channel per color red-green-blue)

def test_2d_conv():
        # quick debug
    input_data = torch.rand(3, 10, 10)  # Example input data with 3 channels and img size 10x10
    filters = [torch.rand(3, 3, 3) for _ in range(2)]
    output = twod_cnn.apply_2D_convolution(input_data, filters, stride=2)

    out_len = (input_data.shape[1] - filters[0].shape[1]) // 2 + 1
    out_wid = (input_data.shape[2] - filters[0].shape[2]) // 2 + 1

    print(output.shape[1] == out_len) # Should be True
    print(output.shape[2] == out_wid) # Should be True
    print("Expected output length:", out_len)
    print("Expected output width:", out_wid)
    print("Output shape:", output.shape) # Should be (number of filters, output length, output width)
    print("Output:", output) # Display the output array


def test_2d_pool():
    # quick debug
    input_data = torch.rand(2, 10, 10)  # Example input
    pool_size = 3
    stride = 2
    output = twod_cnn.apply_2D_pooling(input_data, pool_size, stride)
    expected_out_len = (input_data.shape[1] - pool_size) // stride + 1
    expected_out_wid = (input_data.shape[2] - pool_size) // stride + 1
    print(output.shape[1] == expected_out_len)  # Should be True
    print(output.shape[2] == expected_out_wid)  # Should be True
    print("Expected output length:", expected_out_len)
    print("Expected output width:", expected_out_wid)
    print("Output shape:", output.shape)  # Should be (num_channels, output_length, output_width)
    print("Output:", output)  # Display the output array

#visualize_image()
test_2d_pool()





def see_2d_CNN():
    img_path = 'Lenna_(test_image).png'
    img = io.imread(img_path)
    x, y = 100, 150
    patch = img[y:y+5, x:x+5, :]
    img_gray = 0.299 * patch[:,:,0] + 0.587 * patch[:,:,1] + 0.114 * patch[:,:,2]
    img_gray_3d = img_gray[np.newaxis, :,:]
    output = twod_cnn.two_dimensional_cnn(img_gray_3d, )

    # show the final output
    plt.figure()
    plt.title('Final Output of 2D CNN')
    plt.imshow(output[0], cmap='gray')  # Index [0] to get first
    plt.axis('off')
    plt.show()