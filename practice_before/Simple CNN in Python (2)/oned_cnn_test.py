
import torch 
from oned_cnn import one_dimensional_cnn
# Example Usage
input_data = torch.tensor([[1, 5, 3, 7, 5, 3, 5, 6, 9, 1]])  # Example 1D input
filters_layer1 = [torch.tensor([[1, -1, 1]]),
                  torch.tensor([[1, -3, 2]])]
filters_layer2 = [torch.tensor([[-2, 1], [2, -1]])]
pool_size = 2

output = one_dimensional_cnn(input_data, filters_layer1, filters_layer2, pool_size)
print("Final Output:", output)