import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
from Model import Model
from Camera import Camera
import RPi.GPIO as GPIO
from dependencies.MFRC522_python.mfrc522.SimpleMFRC522 import SimpleMFRC522
import microbit

class View:
    def __init__(self):
        self.tasks = []
        self.model = Model()
        self.camera = Camera()
    
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

        # if self.model.check_badge(self.idBadge, self.username) :
        #     messagebox.showinfo("RFID Scan Successful", "Card ID: " + str(self.idBadge))
            self.afterPreLogin() 
        # else:
        #     messagebox.showerror("RFID Scan Failed", "you don't have the right card")
            


    # 
    # METHODES vues
    # 
            
    def microBit(self, result):
        if result:
            microbit.display.show(Image.YES)
            microbit.sleep(5000)
            microbit.display.clear()
        else:
            microbit.display.show(Image.NO)
            microbit.sleep(5000)
            microbit.display.clear()
             

    def destroy_widgets(self):
        for widget in self.root.winfo_children():
            widget.destroy()

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
        # rfid_prompt_window = tk.Toplevel()
        # rfid_prompt_window.title("RFID Scanning Prompt")

        # rfid_label = tk.Label(rfid_prompt_window, text="Hold an RFID card near the reader.")
        # rfid_label.pack(pady=10)

        # # recup l'id du badge et check via l'api si il est bon
        # try:
        #     reader = SimpleMFRC522()
        #     rfid_prompt_window.update_idletasks()
        #     rfid_prompt_window.update()

        #     self.idBadge, text = reader.read()
        self.check_badge()
           

        # except Exception as e:
        #     messagebox.showerror("RFID Scan Error", "Error during RFID scan: " + str(e))
        # finally:
        #     rfid_prompt_window.destroy()



    def afterPreLogin(self):
        self.destroy_widgets()
        result = self.camera.ReconnaissanceFacial(self.username)

        if result:   
            messagebox.showinfo("LOGIN SUCCESFULL","LOGIN SUCCESFULL")
            self.destroy_widgets()          
            self.microBit(True)
             


    def update(self):
        ret, frame = self.vid.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.root.after(10, self.update)
            

        