import tkinter as tk
from tkinter import filedialog

from algorithms.FPGrowth.a import FPTree
from algorithms.FPGrowth.b import FPTreeGUI
from algorithms.apriori import SystemData, load_and_preprocess_data, update_system_data, main as apriori_main, \
    find_frequent_itemsets
from algorithms.cart import main as cart_main
from algorithms.knn import main as knn_main
from algorithms.naive_bayes import main as naive_bayes_main
from algorithms.svm import main as svm_main

job_options = ["admin", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed",
               "services", "student", "technician", "unemployed", "unknown"]
marital_options = ["divorced", "single", "married"]
education_options = ["unknown", "primary", "secondary", "tertiary"]
default_options = ["no", "yes"]
housing_options = ["no", "yes"]
loan_options = ["no", "yes"]
poutcome_options = ["failure", "success", "other", "unknown"]


def center_window(root, width=300, height=200):
    # get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # calculate position x and y coordinates
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2) - 100
    root.geometry('%dx%d+%d+%d' % (width, height, x, y))


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Giải thuật học máy")
    center_window(root, 800, 600)


    def hide_widgets(widget_list):
        for widget in widget_list:
            widget.grid_remove()


    def show_widgets(widget_list):
        for widget in widget_list:
            widget.grid()


    ALGORITHM_VARIABLES = {
        "K-Nearest Neighbor": {
            "Chiều dài đài hoa (cm) [Số]": tk.DoubleVar(),
            "Chiều rộng đài hoa (cm) [Số]": tk.DoubleVar(),
            "Chiều dài cánh hoa (cm) [Số]": tk.DoubleVar(),
            "Chiều rộng cánh hoa (cm) [Số]": tk.DoubleVar()
        },
        "Cart": {
            'Age': tk.IntVar(),
            'Job': tk.StringVar(value=job_options[0]),
            'Marital': tk.StringVar(value=marital_options[0]),
            'Education': tk.StringVar(value=education_options[0]),
            'Default': tk.StringVar(value=default_options[0]),
            'Balance': tk.IntVar(),
            'Housing': tk.StringVar(value=housing_options[0]),
            'Loan': tk.StringVar(value=loan_options[0]),
            'Campaign': tk.IntVar(),
        },
        "Naive Bayes": {
            "Email cần phân loại:": tk.StringVar()
        },
        "Apriori": {
            "Enter a product or itemset (comma-separated):": tk.StringVar()
        },
        "FPTree": {
            "Enter ingredients (comma-separated):": tk.StringVar()
        },
        "Support Vector Machine": {
            "Thời gian": tk.DoubleVar(),
            "v1": tk.DoubleVar(),
            "v2": tk.DoubleVar(),
            "v3": tk.DoubleVar(),
            "v4": tk.DoubleVar(),
            "v5": tk.DoubleVar(),
            "v6": tk.DoubleVar(),
            "v7": tk.DoubleVar(),
            "v8": tk.DoubleVar(),
            "v9": tk.DoubleVar(),
            "v10": tk.DoubleVar(),
            "v11": tk.DoubleVar(),
            "v12": tk.DoubleVar(),
            "v13": tk.DoubleVar(),
            "v14": tk.DoubleVar(),
            "v15": tk.DoubleVar(),
            "v16": tk.DoubleVar(),
            "v17": tk.DoubleVar(),
            "v18": tk.DoubleVar(),
            "v19": tk.DoubleVar(),
            "v20": tk.DoubleVar(),
            "v21": tk.DoubleVar(),
            "v22": tk.DoubleVar(),
            "v23": tk.DoubleVar(),
            "v24": tk.DoubleVar(),
            "v25": tk.DoubleVar(),
            "v26": tk.DoubleVar(),
            "v27": tk.DoubleVar(),
            "v28": tk.DoubleVar(),
            "Lượng tiền": tk.DoubleVar(),
        },
    }

    ALGORITHM_FUNCTIONS = {
        "K-Nearest Neighbor": knn_main,
        "Naive Bayes": naive_bayes_main,
        "Cart": cart_main,
        "Apriori": apriori_main,
        "FPTree": FPTree,
        "Support Vector Machine": svm_main
    }


    def create_input_fields(content, active_algorithm):
        hide_widgets(content.winfo_children())
        if not active_algorithm:
            return None

        variables = ALGORITHM_VARIABLES.get(active_algorithm, {})
        if active_algorithm == 'Cart':
            row = 0
            for name, variable in variables.items():
                label = tk.Label(content, text=name)
                label.grid(row=row, column=0, pady=10)

                if name in ['Job', 'Marital', 'Education', 'Default', 'Housing', 'Loan']:
                    options = globals()[name.lower() + '_options']
                    dropdown = tk.OptionMenu(content, variable, *options)
                    dropdown.grid(row=row, column=1, pady=10)
                else:
                    entry = tk.Entry(content, textvariable=variable)
                    entry.grid(row=row, column=1, pady=10)
                row += 1

            button = tk.Button(content, text="Dự đoán",
                               command=lambda: run_algorithm(content, active_algorithm, variables))
            button.grid(row=row, column=0, columnspan=2, pady=10)

        elif active_algorithm == 'Apriori':
            system_data = SystemData()
            initial_data_path = "datasets/bread_basket.csv"
            initial_data = load_and_preprocess_data(initial_data_path)

            if initial_data is not None:
                update_system_data(system_data, initial_data)
            else:
                print("Error loading initial data.")

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

            def show_frequent_itemsets():
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

            input_label = tk.Label(content, text="Enter a product or itemset (comma-separated):")
            input_label.grid(row=0, column=0, columnspan=2, pady=10)

            input_entry = tk.Entry(content)
            input_entry.grid(row=1, column=0, columnspan=2, pady=10)

            find_button = tk.Button(content, text="Find Frequent Itemsets", command=show_frequent_itemsets)
            find_button.grid(row=2, column=0, columnspan=2, pady=10)

            load_data_button = tk.Button(content, text="Load New Dataset", command=lambda: load_new_data(system_data))
            load_data_button.grid(row=3, column=0, columnspan=2, pady=10)

            result_label = tk.Label(content, text="", wraplength=300)
            result_label.grid(row=4, column=0, columnspan=2, pady=10)

            frequent_itemsets_text = tk.Text(content, height=30, width=60, state=tk.DISABLED)
            frequent_itemsets_text.grid(row=5, column=0, columnspan=2, pady=10)

        elif active_algorithm == 'FPTree':
            fptree_gui = FPTreeGUI(content)
            fptree_gui.grid(row=0, column=0, columnspan=2, pady=10)

        else:
            row = 0
            for name, variable in variables.items():
                label = tk.Label(content, text=name)
                label.grid(row=row, column=0, pady=10)

                entry = tk.Entry(content, textvariable=variable)
                entry.grid(row=row, column=1, pady=10)
                row += 1

            button_text = "Dự đoán" if active_algorithm in ["K-Nearest Neighbor", "Support Vector Machine"] else "Chạy"
            button = tk.Button(content, text=button_text,
                               command=lambda: run_algorithm(content, active_algorithm, variables))
            button.grid(row=row, column=0, columnspan=2, pady=10)


    def run_algorithm(content, active_algorithm, variables):
        label = tk.Label(content, text="Đang chạy...")
        label.grid(row=len(variables) + 2, column=0, columnspan=2, pady=10)

        algorithm_function = ALGORITHM_FUNCTIONS.get(active_algorithm)
        if algorithm_function:
            inputs = [variable.get() for variable in variables.values()]
            result = algorithm_function(sample_data=inputs)
            label = tk.Label(content, text="Kết quả: {}".format(result))
            label.grid(row=len(variables) + 3, column=0, columnspan=2, pady=10)


    def create_sidebar(root):
        """Create the sidebar with buttons for each algorithm."""
        sidebar = tk.Frame(root, bg="#6272a4", width=300, height=600)
        sidebar.pack(side="left", fill="y")

        for name, algorithm in ALGORITHM_VARIABLES.items():
            button = tk.Button(sidebar, text=name,
                               command=lambda n=name, a=algorithm: create_input_fields(content_frame, n))
            button.config(width=20, height=2)
            button.pack(pady=10, fill="x")
        return sidebar


    def create_content_window(root):
        """Create the content window with a scrollbar."""
        content = tk.Frame(root, width=500, height=600)
        content.pack(side="right", fill="both", expand=True)

        # Create a canvas inside the content frame
        content_canvas = tk.Canvas(content)
        content_canvas.pack(side="left", fill="both", expand=True)

        # Create a scrollbar and attach it to the canvas
        scrollbar = tk.Scrollbar(content, command=content_canvas.yview)
        scrollbar.pack(side="right", fill="y")

        # Configure the canvas to use the scrollbar
        content_canvas.configure(yscrollcommand=scrollbar.set)

        # Create a frame inside the canvas to hold your widgets
        content_frame = tk.Frame(content_canvas)
        content_canvas.create_window((0, 0), window=content_frame, anchor="nw")

        # Update the scrollregion of the canvas when the size of the frame changes
        content_frame.bind("<Configure>",
                           lambda event: content_canvas.configure(scrollregion=content_canvas.bbox("all")))

        return content_frame


    sidebar = create_sidebar(root)
    content_frame = create_content_window(root)
    root.mainloop()
