# import cv2
# from picamera2 import Picamera2

# # Load the pre-trained Haar Cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier('../hash.xml')

# picam2 = Picamera2()
# picam2.preview_configuration.main.size = (1280, 720)
# picam2.preview_configuration.main.format = "RGB888"
# picam2.preview_configuration.align()
# picam2.configure("preview")
# picam2.start()

# try:
#     while True:
#         # Capture the camera image
#         im = picam2.capture_array()

#         # Convert the image to grayscale for face detection
#         gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

#         # Perform face detection
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

#         # Draw rectangles around the detected faces
#         for (x, y, w, h) in faces:
#             cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)

#         # Display the combined image with both raw camera and face detection
#         cv2.imshow("Camera with Face Detection", im)

#         # Break the loop when 'q' is pressed
#         if cv2.waitKey(1) == ord('q'):
#             break

# finally:
#     # Release resources
#     cv2.destroyAllWindows()
#     picam2.stop()
#     picam2.close()



import cv2
import numpy as np
from picamera2 import Picamera2

def capture_images_and_train(output_xml_file):
    # Initialize variables
    (images, labels, label_id) = ([], [], 0)

    # Create face recognition model
    model = cv2.face.LBPHFaceRecognizer_create()

    # Set up the face detector
    face_cascade = cv2.CascadeClassifier('../hash.xml')

    # Set up the camera
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "BGR888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    # Capture 30 images for training
    images_to_capture = 30
    images_captured = 0

    try:
        while images_captured < images_to_capture:
            # Capture a frame from the camera
            frame = picam2.capture_array()

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                # Crop the face region
                face_region = gray[y:y+h, x:x+w]

                # Resize the face region to a standard size
                face_region_resized = cv2.resize(face_region, (112, 92))

                # Display the captured face region
                cv2.imshow('Captured Face', face_region_resized)

                # Append the face region to the images list
                images.append(face_region_resized)

                # Append the label to the labels list
                labels.append(label_id)

                images_captured += 1

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release resources
        cv2.destroyAllWindows()
        picam2.stop()
        picam2.close()

    # Convert lists to numpy arrays
    (images, labels) = [np.array(lis) for lis in [images, labels]]

    # Train the face recognition model
    model.train(images, labels)

    # Save the trained model to an XML file
    model.save(output_xml_file)

if __name__ == "__main__":
    output_xml_file = 'trained_model.xml'
    capture_images_and_train(output_xml_file)
