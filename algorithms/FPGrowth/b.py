import tkinter as tk
from tkinter import ttk

from algorithms.FPGrowth.a import FPTree, top_receipes_ingredients


class FPTreeGUI(tk.Frame):
    def __init__(self, root):
        super().__init__(root)

        # Create the FPTree object
        self.fptree = FPTree(top_receipes_ingredients, min_sup=4)

        # Create the GUI elements
        self.label = tk.Label(self, text="Enter ingredients:")
        self.text_area = tk.Text(self, height=20, width=50)
        self.predict_button = ttk.Button(self, text="Chạy", command=self.predict)
        self.result_label = tk.Label(self, text="Kết quả:")

        # Place the GUI elements on the grid
        self.label.grid(row=0, column=0, pady=10)
        self.text_area.grid(row=1, column=0, padx=10, pady=10)
        self.predict_button.grid(row=2, column=0, pady=10)
        self.result_label.grid(row=3, column=0, pady=10)

    def predict(self):
        # Get the ingredients from the text area
        user_input = self.text_area.get("1.0", "end-1c")
        user_ingredients = [ingredient.strip().lower() for ingredient in user_input.split(",")]

        # Find the frequent itemsets that contain the user ingredients
        try:
            prefix_path = self.fptree.find_prefix_path(self.fptree.header[user_ingredients[0]][1])
            for ingredient in user_ingredients[1:]:
                prefix_path = self.fptree.find_prefix_path(self.fptree.header[ingredient][1]) & prefix_path

            # Display the results
            if len(prefix_path) == 0:
                self.result_label.config(text="No frequent itemsets found.")
            else:
                result = "Frequent itemsets that contain the user ingredients:"
                for key, value in prefix_path.items():
                    result += "\n"
                    result += f"Items: {', '.join(list(key))}"
                self.result_label.config(text=result)
        except KeyError:
            self.result_label.config(text="No frequent itemsets found.")


if __name__ == "__main__":
    # Create and run the GUI
    root = tk.Tk()
    gui = FPTreeGUI(root)
    root.mainloop()
