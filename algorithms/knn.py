import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main(path="datasets/iris.csv", sample_data=[]):
    # Load the Iris dataset from a local CSV file
    iris_df = pd.read_csv(path)

    # Predict the sample data
    X = iris_df.drop("Species", axis=1)
    X = X.drop("Id", axis=1)
    # Convert sample data to pandas DataFrame
    x_sample_data = pd.DataFrame([sample_data], columns=X.columns)

    y = iris_df["Species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    knn = KNeighborsClassifier(n_neighbors=3, p=2, metric="minkowski")
    knn.fit(X_train, y_train)

    y_pred = knn.predict(x_sample_data)
    return y_pred[0]


if __name__ == "__main__":
    main()
