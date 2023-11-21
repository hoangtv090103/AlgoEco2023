import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, \
    average_precision_score, roc_auc_score, recall_score  # some scoring functions
from sklearn.model_selection import train_test_split  # Cross validation tools, and a train/test split utility
from sklearn.svm import SVC


def main(path="datasets/creditcard.csv", sample_data=[]):
    # Load the sVM dataset from a local CSV file
    df = pd.read_csv(path, delimiter=',')

    X = df.iloc[:, 0:30]
    x_sample_data = pd.DataFrame([sample_data], columns=X.columns)
    y = df.iloc[:, 30:31]
    X.head(), y.head()

    # Split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    training_fraud = sum(y_train.values)
    training_fraud_pct = sum(y_train.values) / len(y_train.values) * 100
    test_fraud = sum(y_test.values)
    test_fraud_pct = sum(y_test.values) / len(y_test.values) * 100
    print(
        "X train: {}\nX test:  {}\ny_train: {}\ny test:  {} \nFraud in train set: {},   {:2f}%\nFraud in test set:  {},  {:2f}%\n".format(
            X_train.shape,
            X_test.shape,
            y_train.shape,
            y_test.shape,
            training_fraud[0], training_fraud_pct[0],
            test_fraud[0], test_fraud_pct[0]))

    # Train using SVM
    clf = SVC(kernel='rbf', gamma='scale', class_weight='balanced')

    clf.fit(X_train, y_train.values.ravel())

    # Predict the sample data
    y_pred = clf.predict(x_sample_data)

    def print_scores(y_t, y_p):
        print(f'Accuracy  :{accuracy_score(y_t, y_p):.2f}')
        print(f'Balanced  :{balanced_accuracy_score(y_t, y_p):.2f}')
        print(f'F1        :{f1_score(y_t, y_p):.2f}')
        print(f'Precision :{precision_score(y_t, y_p):.2f}')
        print(f'Recall    :{recall_score(y_t, y_p):.2f}')
        print(f'roc auc   :{roc_auc_score(y_t, y_p):.2f}')
        print(f'pr)auc    :{average_precision_score(y_t, y_p):.2f}')

    print_scores(y_test, y_pred)
    return y_pred[0]


if __name__ == "__main__":
    main()
