import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Bước 1: Đọc và Chuẩn bị Dữ liệu
path = 'datasets/train.csv'
df = pd.read_csv(path)

# Assume your dataset has a structure like:
# | Món ăn | Nguyên liệu 1 | Nguyên liệu 2 | ... | Nguyên liệu n |
# | -------| --------------| --------------|-----| --------------|
# | Món 1  | 1             | 0             | ... | 1             |
# | Món 2  | 0             | 1             | ... | 0             |
# ...

# Bước 2: Chia Dữ liệu
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

# Bước 3: Huấn Luyện Mô Hình FP-Growth
te = TransactionEncoder()
te_ary_train = te.fit(X_train.values[:, 1:]).transform(X_train.values[:, 1:])
df_encoded_train = pd.DataFrame(te_ary_train, columns=te.columns_)

frequent_itemsets_train = fpgrowth(df_encoded_train, min_support=0.01, use_colnames=True)

# Bước 4: Đánh Giá Mô Hình (Tùy Chọn)
# (Bạn có thể sử dụng tập kiểm tra để đánh giá mô hình)

# Chọn một số lượng mẫu tương đương với số lượng dự đoán
num_samples = len(frequent_itemsets_train)
predictions_train = frequent_itemsets_train.apply(lambda row: any(row['itemsets'].issubset(set(x)) for x in X_test.values[:, 1:]), axis=1)

accuracy_train = accuracy_score(X_test.any(axis=1)[:num_samples], predictions_train)
print("Độ chính xác trên tập kiểm tra:", accuracy_train)
