import tkinter as tk

from algorithms import ALGORITHMS

if __name__ == "__main__":
    variables = {}


    def create_input_fields(content, active_algorithm):
        for widget in content.winfo_children():
            widget.destroy()
        if not active_algorithm:
            return None
        if active_algorithm == "K-Nearest Neighbor":
            variables.update({
                "Chiều dài đài hoa (cm) [Số]": tk.DoubleVar(),
                "Chiều rộng đài hoa (cm) [Số]": tk.DoubleVar(),
                "Chiều dài cánh hoa (cm) [Số]": tk.DoubleVar(),
                "Chiều rộng cánh hoa (cm) [Số]": tk.DoubleVar(),
            })
            row = 0
            for name, variable in variables.items():
                label = tk.Label(content, text=name)
                label.grid(row=row, column=0, pady=10)

                entry = tk.Entry(content, textvariable=variable)
                entry.grid(row=row, column=1, pady=10)
                row += 1
            button = tk.Button(content, text="Dự đoán",
                               command=lambda: run_algorithm(content, active_algorithm))
            button.grid(row=row, column=0, columnspan=2, pady=10)


    def run_algorithm(content, active_algorithm):
        if active_algorithm == "K-Nearest Neighbor":
            from algorithms.knn import main
            sample_data = [variable.get() for variable in variables.values()]
            result = main(sample_data=sample_data)
            label = tk.Label(content, text="Kết quả: {}".format(result))
            label.grid(row=len(variables) + 2, column=0, columnspan=2, pady=10)


    root = tk.Tk()
    active_algorithm = tk.StringVar()
    root.title("Giải thuật học máy")
    # Căn giữa màn hình
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = int((screen_width / 2) - (800 / 2))
    y = int((screen_height / 2) - (600 / 2))
    root.geometry("800x600+{}+{}".format(x, y))

    # Tạo Sidebar
    sidebar = tk.Frame(root, bg="#6272a4", width=300, height=600)
    sidebar.pack(side="left", fill="y")

    for name, algorithm in ALGORITHMS.items():
        button = tk.Button(sidebar, text=name,
                           command=lambda n=name, a=algorithm: create_input_fields(content, n))
        button.pack(pady=10)

    # Tạo ContentWindow
    content = tk.Frame(root, width=500, height=600)
    content.pack(side="right", fill="both", expand=True)
    root.mainloop()
