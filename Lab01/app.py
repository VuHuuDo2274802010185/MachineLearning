from flask import Flask, render_template

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Định nghĩa route cho trang chủ
@app.route('/')
def home():
    return render_template('index.html')

# Định nghĩa route cho trang about
@app.route('/about')
def about():
    return "This is the About page"

# Chạy server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
