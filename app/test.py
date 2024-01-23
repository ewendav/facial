import cv2
from picamera2 import Picamera2
import picamera
import picamera.array
import cv2
import numpy as np
import time
# picam2 = Picamera2()
# picam2.preview_configuration.main.size = (1280,720)
# picam2.preview_configuration.main.format = "RGB888"
# picam2.preview_configuration.align()
# picam2.configure("preview")
# picam2.start()

# process_this_frame = True

# while True:
#     video_capture = cv2.VideoCapture(0)
#     ret, frame = video_capture.read()

#     if process_this_frame:
#         small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#         rgb_small_frame = small_frame[:, :, ::-1]
#         face_locations = face_recognition.face_locations(rgb_small_frame)
    
#     process_this_frame = not process_this_frame

#     for (top, right, bottom, left) in face_locations:
#         # Scale back up face locations since the frame we detected in was scaled to 1/4 size
#         top *= 4
#         right *= 4
#         bottom *= 4
#         left *= 4

#         # Draw a box around the face
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#         # Draw a label with a name below the face
#         # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#         # font = cv2.FONT_HERSHEY_DUPLEX
#         # cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

#     cv2.imshow('Video', frame)




# video_capture.release()
# cv2.destroyAllWindows()



face_cascade = cv2.CascadeClassifier('../hash.xml' + 'haarcascade_frontalface_default.xml')

# Initialize the PiCamera
with picamera.PiCamera() as picam:
    # Set preview configuration
    picam.resolution = (640, 480)
    picam.framerate = 30
    picam.rotation = 180  # Rotate if needed

    # Create an array to store the image
    image_array = np.empty((480, 640, 3), dtype=np.uint8)

    try:
        # Capture frames indefinitely
        while True:
            # Capture a single image into the array
            picam.capture(image_array, 'bgr', use_video_port=True)

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