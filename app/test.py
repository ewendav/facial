from picamera2 import Picamera2
import cv2
import numpy as np

# Load the pre-trained Haarcascades face detector
face_cascade = cv2.CascadeClassifier('../hash.xml' + 'haarcascade_frontalface_default.xml')

# Initialize the Picamera2
picamera = Picamera2()

# Set preview configuration
picamera.preview_configuration.main.size = (640, 480)
picamera.preview_configuration.main.format = "rgb"  # Change to "rgb" or "bgr" as needed
picamera.preview_configuration.align()
picamera.configure("preview")
picamera.start()

while True:
    # Capture a single image
    frame = picamera.capture(format='rgb')  # Change to "rgb" or "bgr" as needed

    # Convert the image from RGB to BGR
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)  # Convert to gray
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.2,
                                          minNeighbors=5,
                                          minSize=(80, 80))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the camera
picamera.stop()
cv2.destroyAllWindows()
