import tkinter as tk
from tkinter import messagebox
from Model import Model


class View:
    def __init__(self):
        self.tasks = []
        self.model = Model()

    def check_credentials(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        if self.model.check_credentials(username, password) :
            messagebox.showinfo("Login Successful", "Welcome, " + username + "!")
        else : 
            messagebox.showinfo("Login failled", "Bye" )



    def loginStart(self):        
        # créer fenetre principalle
        root = tk.Tk()
        root.title("Login Form")

        # créer et place les eleemnts 
        username_label = tk.Label(root, text="Username:")
        username_label.grid(row=0, column=0, padx=10, pady=10)

        self.username_entry = tk.Entry(root)
        self.username_entry.grid(row=0, column=1, padx=10, pady=10)

        password_label = tk.Label(root, text="Password:")
        password_label.grid(row=1, column=0, padx=10, pady=10)

        self.password_entry = tk.Entry(root, show="*")
        self.password_entry.grid(row=1, column=1, padx=10, pady=10)

        login_button = tk.Button(root, text="Login", command=self.check_credentials)
        login_button.grid(row=2, column=0, columnspan=2, pady=10)

        # lance la boucle tinker
        root.mainloop()


    