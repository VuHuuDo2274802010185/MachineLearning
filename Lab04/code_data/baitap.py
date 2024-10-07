# Nhập thư viện
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Lấy dữ liệu
data = pd.read_csv(r'C:\Users\vuhuu\OneDrive\Desktop\Ai\VuHuuDo_2274802010185_MachineLearning\MachineLearning\Lab04\code_data\drug200.csv')

# Tạo tập X và y
X = data.drop('Drug', axis=1)  # X là tất cả các cột trừ cột 'Drug'
y = data['Drug']  # y là cột 'Drug'

# Xem giá trị xuất hiện của các cột định tính để biến đổi bước tiếp theo
print("Giá trị của cột Sex:", np.unique(data['Sex']))
print("Giá trị của cột BP:", np.unique(data['BP']))
print("Giá trị của cột Cholesterol:", np.unique(data['Cholesterol']))
print("Giá trị của cột Drug:", np.unique(data['Drug']))

# Biến đổi dữ liệu định tính sang định lượng
# 'M': 0, 'F': 1
X['Sex'] = X['Sex'].map({'M': 0, 'F': 1})

# 'HIGH': 2, 'NORMAL': 1, 'LOW': 0
X['BP'] = X['BP'].map({'HIGH': 2, 'NORMAL': 1, 'LOW': 0})

# 'HIGH': 1, 'NORMAL': 0
X['Cholesterol'] = X['Cholesterol'].map({'HIGH': 1, 'NORMAL': 0})

# 'drugA': 0, 'drugB': 1, 'drugC': 2, 'drugX': 3, 'DrugY': 4
y = y.map({'drugA': 0, 'drugB': 1, 'drugC': 2, 'drugX': 3, 'DrugY': 4})

print(X)
print(y)

# Tạo dữ liệu train/test với tỉ lệ tập test là 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print(y_train)

# Dùng model Decision Tree
decisionTree = DecisionTreeClassifier(min_samples_split=2, max_depth=10)
decisionTree.fit(X_train, y_train)

y_pred = decisionTree.predict(X_test)
print(y_pred)

print(y_test.values)

# Độ chính xác
accuracy = accuracy_score(y_test.values, y_pred)
print("Độ chính xác của Decision Tree:", accuracy)

# Dùng model Random Forest
randomForest = RandomForestClassifier(n_estimators=3, max_features=4)
randomForest.fit(X_train, y_train)

y_pred = randomForest.predict(X_test)
print(y_pred)

print(y_test.values)

# Độ chính xác
accuracy = accuracy_score(y_test.values, y_pred)
print("Độ chính xác của Random Forest:", accuracy)