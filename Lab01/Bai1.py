import numpy as np
import matplotlib.pyplot as plt

# Chiều cao của các bạn sinh viên trong lớp (cm)
X = np.array([180, 162, 183, 174, 160, 163, 180, 165, 175, 170, 170, 169, 168, 175, 169, 171, 155, 158, 175, 165]).reshape(-1, 1)
# Cân nặng của các bạn sinh viên trong lớp (kg)
y = np.array([86, 55, 86.5, 70, 62, 54, 60, 72, 93, 89, 60, 82, 59, 75, 56, 89, 45, 60, 60, 72]).reshape(-1, 1)

# Thêm cột 1 vào X để tính toán hệ số theta
X = np.insert(X, 0, 1, axis=1)

# Tính toán theta theo công thức (X^T * X)^(-1) * X^T * y
theta = np.linalg.inv(X.T @ X) @ X.T @ y

# Vẽ đường hồi quy
x1 = 150
y1 = theta[0] + theta[1] * x1
x2 = 190
y2 = theta[0] + theta[1] * x2
plt.plot([x1, x2], [y1, y2], 'r-')

# Vẽ các điểm dữ liệu
plt.plot(X[:, 1], y, 'bo')
plt.xlabel('Chiều cao (cm)')
plt.ylabel('Cân nặng (kg)')
plt.title('Chiều cao và cân nặng của sinh viên VLU')
plt.show()
