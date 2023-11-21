import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # Import train_test_split
from sklearn.tree import DecisionTreeClassifier

sns.set(rc={'figure.figsize': (14, 10)})
path = 'datasets/train.csv'
df = pd.read_csv(path)
df.head()
# number of ingredients
nr_ingredients = len(df.iloc[:, 0])

features = df.iloc[:, 1:nr_ingredients]
labels = df.iloc[:, 0]

nr_ingredients = len(df.columns) - 1
top_ten_idx = list(features.sum(axis=1).sort_values(ascending=False).index)[:10]

top_receipes_ingredients = [list(features.columns.values[np.where(features.iloc[top_ten_idx[i]] == 1.0)[0]]) for i in
                            range(10)]


class FPTree():
    class Node:
        def __init__(self, name_value, num_occur, parent_node):
            self.name = name_value
            self.count = num_occur
            self.node_link = None
            self.parent = parent_node
            self.children = {}

        def inc(self, num_occur):
            self.count += num_occur

        def disp(self, ind=1):
            print('  ' * ind, self.name, ' ', self.count)
            for child in self.children.values():
                child.disp(ind + 1)

    def __init__(self, data, min_sup):
        self.data = self.init_set(data)
        self.tree, self.header = self.create(self.data, min_sup)

    def create(self, data, min_sup):
        header = {}

        for trans in data:
            for item in trans:
                header[item] = header.get(item, 0) + data[trans]
        for k in list(header):
            if header[k] < min_sup:
                del (header[k])
        freq_set = set(header.keys())

        if len(freq_set) == 0: return None, None
        for k in header:
            header[k] = [header[k], None]

        ret_tree = self.Node('Null Set', 1, None)
        for tranSet, count in data.items():
            local_d = {}
            for item in tranSet:
                if item in freq_set:
                    local_d[item] = header[item][0]
            if len(local_d) > 0:
                ordered_items = [v[0] for v in sorted(local_d.items(), key=lambda p: p[1], reverse=True)]
                self.update(ordered_items, ret_tree, header, count)
        return ret_tree, header

    def update(self, items, tree, header, count):
        if items[0] in tree.children:
            tree.children[items[0]].inc(count)
        else:
            tree.children[items[0]] = self.Node(items[0], count, tree)
            if header[items[0]][1] == None:
                header[items[0]][1] = tree.children[items[0]]
            else:
                self.update_header(header[items[0]][1], tree.children[items[0]])
        if len(items) > 1:
            self.update(items[1::], tree.children[items[0]], header, count)

    def update_header(self, test_node, target_node):
        while (test_node.node_link != None):
            test_node = test_node.node_link
        test_node.node_link = target_node

    def init_set(self, data):
        frozen_set = {}
        for trans in data:
            frozen_set[frozenset(trans)] = 1
        return frozen_set

    def ascend_tree(self, leaf_node, prefix_path):
        if leaf_node.parent != None:
            prefix_path.append(leaf_node.name)
            self.ascend_tree(leaf_node.parent, prefix_path)

    def find_prefix_path(self, tree):
        cond_pats = {}
        while tree != None:
            prefix_path = []
            self.ascend_tree(tree, prefix_path)
            if len(prefix_path) > 1:
                cond_pats[frozenset(prefix_path[1:])] = tree.count
            tree = tree.node_link
        return cond_pats


if __name__ == '__main__':
    fptree = FPTree(top_receipes_ingredients, min_sup=4)
    fptree.tree.disp()
    fptree.find_prefix_path(fptree.header['olive oil'][1])
    fptree.find_prefix_path(fptree.header['apple cider vinegar'][1])

    sns.set(rc={'figure.figsize': (14, 10)})
    path = '../../datasets/train.csv'
    df = pd.read_csv(path)
    # number of ingredients
    nr_ingredients = len(df.iloc[:, 0])

    features = df.iloc[:, 1:nr_ingredients]
    labels = df.iloc[:, 0]

    nr_ingredients = len(df.columns) - 1

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Create the decision tree classifier
    clf = DecisionTreeClassifier()

    # Train the model
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    joblib.dump(fptree, 'fptree_model.joblib')
