import tkinter as tk
import tkinter.ttk as ttk

root = tk.Tk()

# config the root window
root.geometry('500x500')
root.title('PYXIS GUI Widget')

# Distribution type
# Container
dist_type_container = tk.Frame()
dist_type_container.pack()

# Label
dist_type_label = ttk.Label(
    master=dist_type_container, text="Select distribution")
dist_type_label.pack(fill=tk.X, padx=35, pady=5)

# Dropdown menu widget
dist_type_var = tk.StringVar()
dist_type_dropdown = ttk.Combobox(
    dist_type_container, textvariable=dist_type_var)
dist_type_dropdown['values'] = ('Bivariate Normal', 'Random')
dist_type_dropdown['state'] = 'readonly'
dist_type_dropdown.pack(fill=tk.X, padx=1, pady=5)


# Set the value to a variable
def dist_changed(event):
    user_dist = dist_type_var.get()


dist_type_dropdown.bind('<<ComboboxSelected>>', dist_changed)


# pack frames
dist_type_container.pack()
root.mainloop()
