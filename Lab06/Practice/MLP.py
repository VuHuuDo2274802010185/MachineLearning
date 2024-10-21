import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import SGD
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = (0.5, ), std = (0.5, ))])
trainset = torchvision.datasets.FashionMNIST(root='C:/Users/vuhuu/Downloads/DecisionTree_RandomForest', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, num_workers=8, shuffle=True)
testset = torchvision.datasets.FashionMNIST(root='C:/Users/vuhuu/Downloads/DecisionTree_RandomForest', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1024, num_workers=8, shuffle=False)

"""
+ Giải thích các tham số trong class FashionMNIST. Bộ dữ liệu FashionMNIST gồm 60000 tập train và 10000 tập test
root: đường dẫn lưu trữ dữ liệu. Dữ liệu được tải về sẽ lưu tại đây
num_worker: số lượng worker
train: Nếu bạn muốn lấy dữ liệu tập train thì để là True và ngược lại False là tập test
download: tải bộ dữ liệu về
transform: tham số này dùng để chuẩn hóa dữ liệu từ (0 - 255) về (0 - 1). Nếu không chuẩn hóa dữ liệu thì model sẽ không thể train tốt được.

+ Giải thích về class DataLoader.
Sau khi tải tập train và test về thì train giờ sẽ là 60000 ảnh và 10000 cho test. Sau đó chúng ta sẽ đóng 60000 ảnh train và 10000 ảnh test vào block.
batch_size: số lượng ảnh trong 1 block. Với ví dụ trên thì mình sẽ đóng 1024 ảnh vào 1 block => Như vậy chúng ta sẽ có xấp xỉ 58 block 1024 ảnh và số lượng ảnh còn dư lại là block 59.
shuffle: Nếu ta muốn xáo trộn dữ liệu mỗi lần chạy lại thì set là True và ngược lại

"""

def imshow(img):
    img = img * 0.5 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()

for i, (images, labels) in enumerate(trainloader, 0):
    imshow(torchvision.utils.make_grid(images[:8]))
    break
'''
+ Hàm này là dùng để visualize 8 ảnh đầu tiên của 1 block lên. Như mình đã nói ở trên thì giờ train loader của mình sẽ có tầm khoảng 59 block. Sau đó chúng ta
lấy 8 ảnh trong block đầu tiên ra để hiển thị.
'''

def getModel(n_features):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_features, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)
    return model

'''
+ Hàm này dùng để build model. Thì như các bạn đã học về MLP thì đầu tiên chúng ta sẽ làm phẳng ảnh bằng Flatten xong sau đó đưa qua lớp Linear rồi qua hàm kích hoạt Relu và sau cùng
lớp output ở lớp Linear cuối cùng là số lượng class trong bài toán (ở bài toán này là 10). Các bạn có thể nhìn hình dưới đây để hiểu rõ hơn về kiến trúc của MLP.
+ Giải thích thêm ở lớp Linear thì trong lớp Linear sẽ tương tự như việc lan truyền thẳng. Sẽ có 2 trọng số là weight (w) và bias (b). Hai tham số này có lẽ đã rất quen trong thuật toán
Linear Regression. Sau đó nó sẽ tính toán tương tự Linear rồi kết quả của nó sẽ nằm ở node tiếp của layer tiếp theo.
'''

n_features = 28 * 28 # Đối với ảnh trong tập FashionMNIST thì size ảnh là 28x28 => Số lượng feature trong ảnh là 28 * 28 = 784
model = getModel(n_features)
lr = 0.01
optim = SGD(params = model.parameters(), lr = lr)
loss_fn = nn.CrossEntropyLoss()
model
'''
+ Ở đoạn code cell này mình sẽ tạo model, phương thức optimizer network và loss function. Một tham số cũng khá quan trọng là lr (learning rate).
+ Về phương thức optimizer thì sẽ có vài phương thức phổ biến như Adam, SGD, AdamW, ... Thì trong bài toán này mình sẽ sử dụng SGD. (các bạn có thể lên mạng để tìm hiểu thêm về SGD).
+ Hàm loss function mình sử dụng trong đây là CrossEntropyLoss. Thực tế cũng có các hàm loss khác như BCELoss (được sử dụng trong thuật toán Logistic Regression), MSELoss. Các bạn cũng có
thể tìm hiểu thêm về công thức tính toán của nó.
'''

def evaluate(model, testloader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad(): # hàm này là để giữ nguyên giá trị đạo hàm thường phải có khi ta muốn đánh giá model.
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_loss = test_loss / len(testloader)
    return test_loss, accuracy

n_epochs = 10
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        
        inputs, labels = inputs.to(device), labels.to(device)

        optim.zero_grad() # khởi tạo giá trị đạo hàm = 0


        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()


        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        running_correct += (predicted == labels).sum().item()

        # 2 dòng dưới đây là lan truyền ngược và optimizer tham số w và b.
        loss.backward()
        optim.step()        

    epoch_accuracy = 100 * running_correct / total
    epoch_loss = running_loss / (i + 1)
    test_loss, test_accuracy = evaluate(model, testloader, loss_fn)
    print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

plt.figure(figsize = (20, 5))
plt.subplot(121)
plt.title('Loss Epochs')
plt.plot(train_losses, label = 'Train Loss')
plt.plot(test_losses, label = 'Test Loss')
plt.legend()
plt.subplot(122)
plt.title('Accuracy Epoch')
plt.plot(test_accuracies, label = 'Test Accuracy')
plt.legend()


'''
Kết quả dưới đây là một kết quả tốt (chưa xét đến accuracy của model). Vì chúng ta có thể thấy train loss và test loss càng qua các epoch càng giảm và ngược lại
accuracy càng tăng. Nếu các đường thẳng mịn như vậy thì cho thấy model đang fit tốt trên tập dữ liệu. Mình chỉ train trên 10 epoch.
'''

# Sau khi model đã được train xong thì chúng ta sẽ lưu trọng số của model lại để có thể gọi lại model mà không cần train lại.
torch.save(model, "MLP_dress.pth")