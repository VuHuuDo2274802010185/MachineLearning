from flask import Flask, render_template
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

@app.route('/')
def index():
    # Xác định đường dẫn đến file CSV
    csv_file_path = os.path.join(os.path.dirname(__file__), 'data', 'drug200.csv')

    # Đọc dữ liệu từ file CSV
    data = pd.read_csv(csv_file_path)

    # Mã hóa các biến phân loại
    label_encoder = LabelEncoder()
    data['Sex'] = label_encoder.fit_transform(data['Sex'])
    data['BP'] = label_encoder.fit_transform(data['BP'])
    data['Cholesterol'] = label_encoder.fit_transform(data['Cholesterol'])

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
    y = data['Drug']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện mô hình Naive Bayes
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Dự đoán và đánh giá mô hình
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=model.classes_)

    return render_template('index.html', accuracy=accuracy * 100, report=report)

if __name__ == '__main__':
    app.run(debug=True)