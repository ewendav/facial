import tkinter as tk
from tkinter import messagebox
from Model import Model
import RPi.GPIO as GPIO
from dependencies.MFRC522_python.mfrc522.SimpleMFRC522 import SimpleMFRC522



class View:
    def __init__(self):
        self.tasks = []
        self.model = Model()
    
    # 
    # methodes services 
    #    
    def check_credentials(self):
        self.username = self.username_entry.get()
        password = self.password_entry.get()

        if self.model.check_credentials(self.username, password) :
            self.start_rfid_scanning()
        else : 
            messagebox.showinfo("Login failled", "wrong username or password" )

    def check_badge(self):

        if self.model.check_badge(self.idBadge, self.username) :
            messagebox.showinfo("RFID Scan Successful", "Card ID: " + str(self.idBadge))
            self.afterLogin() 
        else:
            messagebox.showerror("RFID Scan Failed", "you don't have the right card")
            


    # 
    # METHODES vues
    # 

    def loginStart(self):        
        # créer fenetre principalle
        self.root = tk.Tk()
        self.root.title("Login Form")

        # créer et place les eleemnts 
        username_label = tk.Label(self.root, text="Username:")
        username_label.grid(row=0, column=0, padx=10, pady=10)

        self.username_entry = tk.Entry(self.root)
        self.username_entry.grid(row=0, column=1, padx=10, pady=10)

        password_label = tk.Label(self.root, text="Password:")
        password_label.grid(row=1, column=0, padx=10, pady=10)

        self.password_entry = tk.Entry(self.root, show="*")
        self.password_entry.grid(row=1, column=1, padx=10, pady=10)

        login_button = tk.Button(self.root, text="Login", command=self.check_credentials)
        login_button.grid(row=2, column=0, columnspan=2, pady=10)

        # lance la boucle tinker
        self.root.mainloop()

    def start_rfid_scanning(self):
        rfid_prompt_window = tk.Toplevel()
        rfid_prompt_window.title("RFID Scanning Prompt")

        rfid_label = tk.Label(rfid_prompt_window, text="Hold an RFID card near the reader.")
        rfid_label.pack(pady=10)

        # recup l'id du badge et check via l'api si il est bon
        try:
            reader = SimpleMFRC522()
            rfid_prompt_window.update_idletasks()
            rfid_prompt_window.update()

            self.idBadge, text = reader.read()
            self.check_badge()
           

        except Exception as e:
            messagebox.showerror("RFID Scan Error", "Error during RFID scan: " + str(e))
        finally:
            rfid_prompt_window.destroy()



    def afterLogin(self):
        self.root.reset_window()
        new_label = tk.Label(self.root, text="New Content", font=('Helvetica', 16))
        new_label.place(relx=0.5, rely=0.5, anchor="center")

        new_button = tk.Button(self.root, text="New Button", command=self.new_function)
        new_button.place(relx=0.5, rely=0.7, anchor="center")
        