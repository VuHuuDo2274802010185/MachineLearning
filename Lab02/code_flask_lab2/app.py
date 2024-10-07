from flask import Flask, render_template, request
import pickle

# Khởi tạo Flask
app = Flask(__name__)

# Tải 3 mô hình và vectorizer đã lưu ở file trước
with open("bernoulli_model.pkl", "rb") as bnb_file:
    bnb_model = pickle.load(bnb_file)

with open("multinomial_model.pkl", "rb") as mnb_file:
    mnb_model = pickle.load(mnb_file)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Route trang chủ
@app.route('/')
def home():
    return render_template('index.html')

# Route để xử lý dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sentence = request.form['sentence']
        # Chuyển câu nhập vào thành vector
        X_new = vectorizer.transform([sentence])
        # Dự đoán bằng Bernoulli và Multinomial
        prediction_bernoulli = bnb_model.predict(X_new)[0]
        prediction_multinomial = mnb_model.predict(X_new)[0]
        return render_template('index.html', 
                               sentence=sentence,
                               prediction_bernoulli=prediction_bernoulli,
                               prediction_multinomial=prediction_multinomial)

if __name__ == '__main__':
    app.run(debug=True)
