from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

@app.route('/')
def index():
    # Đọc dữ liệu từ file CSV
    data = pd.read_csv('C:/Users/vuhuu/OneDrive/Desktop/Ai/VuHuuDo_2274802010185/MachineLearning/Lab02/Code/Education.csv')

    # Mã hóa các biến phân loại
    X = data['Text']
    y = data['Label']

    # Chuyển đổi văn bản thành các đặc trưng số
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    # Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra (80% huấn luyện, 20% kiểm tra)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Khởi tạo mô hình Gaussian Naive Bayes
    gnb = GaussianNB()

    # Huấn luyện mô hình trên dữ liệu huấn luyện
    gnb.fit(X_train.toarray(), y_train)

    # Dự đoán trên dữ liệu kiểm tra
    y_pred = gnb.predict(X_test.toarray())

    # Đánh giá hiệu suất của mô hình
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["text", "Label"])

    return render_template('index.html', accuracy=accuracy * 100, report=report)

if __name__ == '__main__':
    app.run(debug=True)