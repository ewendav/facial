from picamera2 import Picamera2
import cv2
import numpy as np
import time

# Initialize the Picamera2
picamera = Picamera2()

# Set preview configuration
picamera.preview_configuration.main.size = (640, 480)
picamera.preview_configuration.main.format = "rgb"
picamera.preview_configuration.align()
picamera.configure("preview")
picamera.start()

# Create an array to store the image
image_array = np.empty((480, 640, 3), dtype=np.uint8)

# Load the pre-trained Haarcascades face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

try:
    # Capture frames indefinitely
    while True:
        # Capture a single image into the array
        picamera.capture(image_array, 'bgr')  # Change 'rgb' to 'bgr'

        # Convert the image to grayscale for face detection
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

        # Draw rectangles around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image_array, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the image with face rectangles
        cv2.imshow('Face Detection', image_array)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Optional: Adjust the sleep time to control the frame rate
        time.sleep(0.1)

finally:
    # Close OpenCV window and release resources when the loop is exited
    cv2.destroyAllWindows()
