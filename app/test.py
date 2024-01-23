from picamera2 import Picamera2
import cv2
import numpy as np
import time
import face_recognition

# Initialize the Picamera2
picamera = Picamera2()

# Set preview configuration
picamera.preview_configuration.main.size = (640, 480)
picamera.preview_configuration.main.format = "bgr"  # Ensure BGR format
picamera.preview_configuration.align()
picamera.configure("preview")
picamera.start()

# Create an array to store the image
image_array = np.empty((480, 640, 3), dtype=np.uint8)

try:
    # Capture frames indefinitely
    while True:
        # Capture a single image into the array
        picamera.capture(image_array, 'bgr')

        # Resize frame of video to 1/4 size for faster face detection processing
        small_frame = cv2.resize(image_array, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR to RGB
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # Draw rectangles around the faces
        for (top, right, bottom, left) in face_locations:
            # Scale the face locations back up since the frame was resized
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a rectangle around the face
            cv2.rectangle(image_array, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the image with face rectangles
        cv2.imshow('Facial Detection', image_array)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Optional: Adjust the sleep time to control the frame rate
        time.sleep(0.1)

finally:
    # Close OpenCV window and release resources when the loop is exited
    cv2.destroyAllWindows()
