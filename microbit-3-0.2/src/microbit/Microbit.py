import serial
import time

class Microbit():
    def __init__(self, PORT : str):
        ser = serial.Serial()
        ser.baudrate = 115200
        ser.port = PORT
        ser.open()
        self.ser = ser

    def readData(self):
        data = str(self.ser.readline())
        data = data.replace(" ", "")
        data = data.replace("'", "")
        data = data.replace("\\r\\n", "")
        data = data.replace("b", "")
        self.data = data      
        return data

    def showString(self, string : str):
        string = f"{string}\n"
        string = bytes(string, "utf-8")
        self.ser.write(string)

    #Input
    def pressed_a(self):
        if self.data == "A":
            return True

    def pressed_b(self):
        if self.data == "B":
            return True

    def tilt_left(self):
        if self.data == "tiltleft":
            return True 

    def tilt_right(self):
        if self.data == "tiltright":
            return True

    def shake(self):
        if self.data == "shake":
            return True

    def pressed_ab(self):
        if self.data == "A+B":
            return True


    def arrow(self, direction : str):  
        direction = f"{direction}\n"
        direction = bytes(direction, "utf-8")
        self.ser.write(direction)