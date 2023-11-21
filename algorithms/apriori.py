import tkinter as tk
from tkinter import filedialog

import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori


class SystemData:
    def __init__(self):
        self.my_basket = None
        self.frequent_items = None
        self.rules = None


def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'Transaction' not in df.columns or 'Item' not in df.columns:
            raise ValueError("Missing required columns in the file.")

        df['Item'] = df['Item'].str.strip().str.lower()
        transactions_str = df.groupby(['Transaction', 'Item'])['Item'].count().reset_index(name='Count')
        my_basket = transactions_str.pivot_table(index='Transaction', columns='Item', values='Count',
                                                 aggfunc='sum').fillna(0)
        return my_basket
    except Exception as e:
        print(f"Error loading and preprocessing data: {e}")
        return None


def update_system_data(system_data, input_data):
    system_data.my_basket = pd.concat([system_data.my_basket, input_data]).groupby(level=0).sum()
    system_data.frequent_items = apriori(system_data.my_basket.applymap(encode), min_support=0.001, use_colnames=True)
    system_data.rules = association_rules(system_data.frequent_items, metric="lift", min_threshold=1)
    system_data.rules.sort_values('confidence', ascending=False, inplace=True)
    return system_data


def encode(x):
    return 1 if x >= 1 else 0


def find_frequent_itemsets(input_text, system_data):
    if system_data.rules is None:
        print("System data rules have not been initialized.")
        return []

    input_items = [item.strip() for item in input_text.split(',')]
    frequent_itemsets = []
    for index, row in system_data.rules.iterrows():
        antecedents = set(row['antecedents'])
        consequents = set(row['consequents'])
        if all(item in antecedents for item in input_items):
            frequent_itemsets.append((consequents, row['support'], row['confidence'], row['lift']))
    return frequent_itemsets


def show_frequent_itemsets(input_entry, result_label, frequent_itemsets_text):
    global system_data
    input_text = input_entry.get().strip().lower()
    frequent_itemsets = find_frequent_itemsets(input_text, system_data)

    unique_itemsets = set()

    if frequent_itemsets:
        result_label.config(text="Frequent itemsets frequently appearing with {}:".format(input_text))
        frequent_itemsets_text.config(state=tk.NORMAL)
        frequent_itemsets_text.delete(1.0, tk.END)

        for itemsets in frequent_itemsets:
            current_itemset = frozenset(itemsets[0])

            if current_itemset not in unique_itemsets:
                unique_itemsets.add(current_itemset)
                frequent_itemsets_text.insert(tk.END, f"Items: {', '.join(itemsets[0])}\n")

        frequent_itemsets_text.config(state=tk.DISABLED)
    else:
        result_label.config(text="No frequent itemsets found for {}".format(input_text))


def load_new_data(system_data):
    try:
        file_path = filedialog.askopenfilename(title="Chọn file CSV", filetypes=[("CSV files", "*.csv")])
        if file_path:
            new_data = load_and_preprocess_data(file_path)
            if new_data is not None:
                if system_data.my_basket is None:
                    system_data = SystemData()
                system_data = update_system_data(system_data, new_data)
                result_label.config(text="Dữ liệu đã được truyền vào thành công")
                return system_data
            else:
                result_label.config(text="Lỗi truyền dữ liệu vào")
    except Exception as e:
        print(f"Lỗi trong quá trình tải dữ liệu: {e}")


def main(data_path="datasets/bread_basket.csv"):
    system_data = SystemData()
    initial_data = load_and_preprocess_data(data_path)

    if initial_data is not None:
        update_system_data(system_data, initial_data)
    else:
        print("Error loading initial data.")
    return system_data.frequent_items


if __name__ == "__main__":
    main()
    root = tk.Tk()
    root.title("Frequent Itemsets Finder")

    input_label = tk.Label(root, text="Enter a product or itemset (comma-separated):")
    input_label.pack(pady=10)

    input_entry = tk.Entry(root)
    input_entry.pack()

    find_button = tk.Button(root, text="Find Frequent Itemsets", command=show_frequent_itemsets)
    find_button.pack(pady=10)
    load_data_button = tk.Button(root, text="Load New Dataset", command=lambda: load_new_data(system_data))
    load_data_button.pack(pady=10)

    result_label = tk.Label(root, text="", wraplength=300)
    result_label.pack()

    frequent_itemsets_text = tk.Text(root, height=30, width=60, state=tk.DISABLED)
    frequent_itemsets_text.pack()

    root.mainloop()
