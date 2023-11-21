# CART với dict
# dict
import csv

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier


def main(path="datasets/iris.csv", sample_data=[]):
    def read_bank_dataset():
        dataset = []
        with open('datasets/bank-full.csv', newline='') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            is_first = True
            for row in reader:
                instance = {}
                if not is_first:
                    instance["age"] = float(row[0])

                    job_dict = {"admin.": 0, "blue-collar": 1, "entrepreneur": 2, "housemaid": 3, "management": 4,
                                "retired": 5, "self-employed": 6, "services": 7, "student": 8, "technician": 9,
                                "unemployed": 10, "unknown": 11}
                    instance["job"] = float(job_dict[row[1]])

                    marital_dict = {"divorced": 0, "single": 1, "married": 2}
                    instance["marital"] = float(marital_dict[row[2]])

                    education_dict = {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}
                    instance["education"] = float(education_dict[row[3]])

                    instance["default"] = 1 if row[4] == "yes" else 0
                    instance["balance"] = float(row[5])
                    instance["housing"] = 1 if row[6] == "yes" else 0
                    instance["loan"] = 1 if row[7] == "yes" else 0

                    instance["campaign"] = float(row[12])

                    instance["previous"] = float(row[14])

                    poutcome_dict = {"failure": 0, "success": 1, "other": 2, "unknown": 3}
                    instance["poutcome"] = float(poutcome_dict[row[15]])

                    instance["y"] = row[16]

                    dataset.append(instance)
                is_first = False
        return dataset

    # Đọc dữ liệu từ tệp CSV
    data = read_bank_dataset()

    # Chọn các cột số làm đặc trưng (loại bỏ 'quality' ra khỏi danh sách)
    features = [col for col in data[0].keys() if col != 'y']
    # Tạo tập dữ liệu đầu vào (X) và tập dữ liệu đầu ra (y)
    X = []
    y = []
    for row in data:
        instance = []
        for feature in features:
            instance.append(row[feature])
        X.append(instance)
        y.append(row['y'])

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Xây dựng mô hình cây CART
    # model = DecisionTreeClassifier(criterion='gini')

    # Xác định khoảng giá trị ccp_alpha
    alpha_range = np.linspace(0, 0.05, 100)

    best_alpha = None
    best_accuracy = 0

    # Tính toán CCP alphas
    # ccp_alphas, impurities = model.cost_complexity_pruning_path(X_train, y_train)

    # Huấn luyện và đánh giá mô hình trên mỗi giá trị ccp_alpha
    for ccp_alpha in alpha_range:
        model = DecisionTreeClassifier(criterion='gini', ccp_alpha=ccp_alpha)
        scores = cross_val_score(model, X_train, y_train, cv=5)
        avg_accuracy = np.mean(scores)

        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_alpha = ccp_alpha

    print(f"Best CCP Alpha = {best_alpha:.3f}, Best Average Accuracy = {best_accuracy:.3f}")

    # Sử dụng giá trị ccp_alpha tốt nhất để xây dựng mô hình
    best_model = DecisionTreeClassifier(criterion='gini', ccp_alpha=best_alpha)
    best_model.fit(X_train, y_train)

    # Dự đoán trên tập kiểm tra
    y_pred = best_model.predict(X_test)

    # Độ chính xác
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Báo cáo phân loại
    print(classification_report(y_test, y_pred))

    import joblib

    # Lưu mô hình đã được huấn luyện
    joblib.dump(best_model, 'CART.joblib')

    return y_pred[0]
