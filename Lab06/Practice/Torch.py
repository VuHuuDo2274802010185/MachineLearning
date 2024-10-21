import torch
import torch.nn.functional as F

# Công thức tính CrossEntropy Loss
def crossEntropyLoss(output, target):
    # Sử dụng hàm softmax cho output để chuyển đổi thành xác suất và log, sau đó tính cross-entropy
    loss = -torch.sum(target * torch.log(F.softmax(output, dim=0)))
    return loss

# Công thức tính Mean Square Error
def meanSquareError(output, target):
    loss = torch.mean((output - target) ** 2)
    return loss

# Công thức tính BinaryEntropy Loss
def binaryEntropyLoss(output, target, n):
    # Sử dụng hàm sigmoid cho output và sau đó tính binary cross-entropy
    output = torch.sigmoid(output)
    loss = -torch.sum(target * torch.log(output) + (1 - target) * torch.log(1 - output)) / n
    return loss

inputs = torch.tensor([0.1, 0.3, 0.6, 0.7])
target = torch.tensor([0.31, 0.32, 0.8, 0.2])
n = len(inputs)
mse = meanSquareError(inputs, target)
binary_loss = binaryEntropyLoss(inputs, target, n)
cross_loss = crossEntropyLoss(inputs, target)
print(f"Mean Square Error: {mse}")
print(f"Binary Entropy Loss: {binary_loss}")
print(f"Cross Entropy Loss: {cross_loss}")

# Công thức hàm sigmoid
def sigmoid(x: torch.tensor):
    return 1 / (1 + torch.exp(-x))

# Công thức hàm relu
def relu(x: torch.tensor):
    return torch.maximum(torch.tensor(0.0), x)

# Công thức hàm softmax
def softmax(zi: torch.tensor):
    exps = torch.exp(zi - torch.max(zi))  # Trừ đi giá trị lớn nhất để tránh tràn số.
    return exps / torch.sum(exps)

# Công thức hàm tanh
def tanh(x: torch.tensor):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

x = torch.tensor([1.0, 5.0, -4.0, 3.0, -2.0])
f_sigmoid = sigmoid(x)
f_relu = relu(x)
f_softmax = softmax(x)
f_tanh = tanh(x)
print(f"Sigmoid = {f_sigmoid}")
print(f"Relu = {f_relu}")
print(f"Softmax = {f_softmax}")
print(f"Tanh = {f_tanh}")
