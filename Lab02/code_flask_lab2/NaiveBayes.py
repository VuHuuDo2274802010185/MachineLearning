import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split

# Đọc dữ liệu từ CSV (nhớ chỉnh lại đường dẫn dữ liệu)
data1 = pd.read_csv('Education.csv')

# Tiền xử lý dữ liệu
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data1['Text'])
y = data1['Label']

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Huấn luyện mô hình Bernoulli và Multinomial Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train, y_train)

mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Lưu mô hình và vectorizer (sau khi chạy sẽ tạo ra 3 file .pkl)
with open("bernoulli_model.pkl", "wb") as bnb_file:
    pickle.dump(bnb, bnb_file)

with open("multinomial_model.pkl", "wb") as mnb_file:
    pickle.dump(mnb, mnb_file)

with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)
