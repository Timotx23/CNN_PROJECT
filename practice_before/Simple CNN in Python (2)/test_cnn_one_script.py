import CNN_one_script 
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# quick debug
def check_apply_convelution():
    input_data = torch.rand(3, 10)  # Example input data with 3 channels and length 10
    filters = [torch.rand(3, 3) for _ in range(2)]

    output = CNN_one_script.apply_convolution(input_data, filters, stride=2)

    out_len = (input_data.shape[1] - filters[0].shape[1]) // 2 + 1

    print(output.shape[1] == out_len)  # Should be True
    print("Expected output length:", out_len)
    print("Output shape:", output.shape)  # Should be (number of filters, output length)
    print("Output:", output)
def visualize():    
    

    # Create a simple 1D signal with edges
    signal = torch.tensor([[0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=torch.float32)
    # Shape: (1, 15)

    # Define a simple edge detection filter
    edge_filter = torch.tensor([[-1, 1]], dtype=torch.float32)
    filters = [edge_filter]

    # Apply convolution
    def apply_convolution(signal, filters):
        outputs = []

        # Conv1d expects input shape: (batch, channels, length)
        x = signal.unsqueeze(0)  # (1, 1, 15)

        for f in filters:
            kernel = f.unsqueeze(0)  # (1, 1, 2)
            out = F.conv1d(x, kernel)  # (1, 1, 14)
            outputs.append(out.squeeze(0).squeeze(0))  # (14,)

        return outputs

    output = apply_convolution(signal, filters)

    # Plotting
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original Signal")
    plt.plot(signal[0].numpy(), label="Signal")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Edge Detection Output")
    plt.plot(output[0].numpy(), label="Filtered", color="orange")
    plt.legend()

    plt.show()


def apply_more_viz():


    # Create x-axis
    x = torch.linspace(0, 10, 100)

    # Define three edge types
    # (a) Step edge
    step_edge = torch.where(x < 5, torch.tensor(0.0), torch.tensor(1.0))

    # (b) Ramp edge
    ramp_edge = torch.clamp((x - 3) / 4, 0, 1)

    # (c) Ridge edge (spike)
    ridge_edge = torch.exp(-((x - 5)**2) / 0.5)

    
    def plot_ideal_edges(x, step_edge, ramp_edge, ridge_edge):
        """
        Plots three variants of 1D ideal edges with corresponding intensity profiles.
        (a) Step edge
        (b) Ramp edge
        (c) Ridge edge
        """
        # Convert tensors to numpy for plotting
        x_np = x.numpy()
        step_np = step_edge.numpy()
        ramp_np = ramp_edge.numpy()
        ridge_np = ridge_edge.numpy()

        # Create figure with 2 rows and 3 columns
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        
        # ===== STEP EDGE (a) =====
        gradient_step = step_np.repeat(20).reshape(20, -1)
        axes[0, 0].imshow(gradient_step, cmap='gray', aspect='auto')
        axes[0, 0].set_title('(a) Step Edge', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[1, 0].plot(x_np, step_np, 'b-', linewidth=2.5)
        axes[1, 0].set_xlabel('Position', fontsize=10)
        axes[1, 0].set_ylabel('Intensity', fontsize=10)
        axes[1, 0].set_ylim(-0.1, 1.2)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # ===== RAMP EDGE (b) =====
        gradient_ramp = ramp_np.repeat(20).reshape(20, -1)
        axes[0, 1].imshow(gradient_ramp, cmap='gray', aspect='auto')
        axes[0, 1].set_title('(b) Ramp Edge', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[1, 1].plot(x_np, ramp_np, 'g-', linewidth=2.5)
        axes[1, 1].set_xlabel('Position', fontsize=10)
        axes[1, 1].set_ylabel('Intensity', fontsize=10)
        axes[1, 1].set_ylim(-0.1, 1.2)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # ===== RIDGE EDGE (c) =====
        gradient_ridge = ridge_np.repeat(20).reshape(20, -1)
        axes[0, 2].imshow(gradient_ridge, cmap='gray', aspect='auto')
        axes[0, 2].set_title('(c) Ridge Edge (Line)', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        axes[1, 2].plot(x_np, ridge_np, 'r-', linewidth=2.5)
        axes[1, 2].set_xlabel('Position', fontsize=10)
        axes[1, 2].set_ylabel('Intensity', fontsize=10)
        axes[1, 2].set_ylim(-0.1, 1.2)
        axes[1, 2].grid(True, alpha=0.3)
        
        fig.suptitle('Three Variants of 1D Ideal Edges', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.show()
    plot_ideal_edges(x, step_edge, ramp_edge, ridge_edge)


def test_pool():
    # quick debug
    input_data = torch.rand(2, 15)  # Example input data with 2 channels and length 15
    pool_size = 3
    stride = 2
    output = CNN_one_script.apply_pooling(input_data, pool_size, stride)

    expected_out_len = (input_data.shape[1] - pool_size) // stride + 1

    print(output.shape[1] == expected_out_len)  # Should be True
    print("Expected output length:", expected_out_len)
    print("Output shape:", output.shape)  # Should be (num_channels, output_length)
    print("Output:", output)  # Display the output array

#call_viz = visualize()
#apply_more_viz()
def test_relu():
    # quick debug test

    input_data = torch.tensor([[-1, 2, -3], [4, -5, 6]])
    output = CNN_one_script.apply_relu(input_data)
    print("Input:\n", input_data)
    print("Output after ReLU:\n", output) # all negative values should be 0
test_relu()