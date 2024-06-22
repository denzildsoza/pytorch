import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import system
system('cls')
scalar = torch.tensor(7)
# print(scalar)

vector = torch.tensor([7,7])
# print(vector)

matrix = torch.tensor([[2,7],[4,7]])
# print(matrix)
# print(matrix.shape)
# print(matrix.ndim)
# print(matrix[0])
 

tensor = torch.tensor([[
    [1,2,3],
    [4,5,6],
    [7,8,9],
]])


random_tensors = torch.rand(size=(234,234,3))
# print(random_tensors)

#create tensors in a range of no's
torch.arange(1,11)

#creating similar tensors
torch.ones_like
torch.zeros_like

float_32_tensor=torch.tensor([3.0,4.0,5.0]
                             ,dtype=torch.float64,#tensor datatype
                             device=None,
                             requires_grad=False)#weather to track gradient or not




#operations in tensor include + - * matmul(matrix multiplication)
#to matmul shape -> (3,2)@(2,3) => (3,3)
#to multiply tensors that dont obey rule tensors must have same inner dimensions 
#use transpose method
tensor_a = torch.tensor([[7,8],[8,9],[9,4]])
tensor_b = torch.tensor([[7,8],[8,9],[9,4]])
#tensor_mm = torch.mm(tensor_a,tensor_b)   mat1 and mat2 shapes cannot be multiplied (3x2 and 3x2)
tensor_b_transposed = tensor_b.T
tensor_mm = torch.mm(tensor_a,tensor_b_transposed) 

#min, max,mean,sum, etc (tensor aggregation)
torch.argmin(tensor_a) #returns index of the minimum value
#argmax

##reshaping,view, stacking(vstack,hstack),permute, squeezing and unsqeezing
#check the pytorch oficial docs

# Indexing in pytorch tensor
x= torch.arange(1,10).reshape(1,3,3)
# print(x[0])
# print(x[0][0])
# print(x[0][0][0])
# print(x[:,:,1]) # get all values of 0th and first dimension and give 1st index fo 2nd dimension value


#pytorch tensors and numpy
# numpy is a dependency of pytorch hence we can interact with numpy in pytorch
array = np.arange(1.0,8.0)
tensor = torch.from_numpy(array)


#tensor to numpy
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()


#reproducability
#how neural networks work
# to reproduce randomness in nn's 
random_tensor_a=torch.rand(3,4)
random_tensor_b=torch.rand(3,4)
# print(random_tensor_a==random_tensor_b)

#making reproducable tensors
Random_seed = 42
torch.manual_seed(Random_seed)
random_tensor_c=torch.rand(3,4)
torch.manual_seed(Random_seed)
random_tensor_d=torch.rand(3,4)
print(random_tensor_c==random_tensor_d)
































