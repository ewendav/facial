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

def photoEntrainement(output_xml_file):
    # Initialize three empty lists: one for captured images, one for labels, and one for label IDs
    (images, labels, label_id) = ([], [], 0)

    # Create a LBPH (Local Binary Patterns Histograms) face recognition model
    model = cv2.face.LBPHFaceRecognizer_create()

    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier('../hash.xml')

    # Create an instance of the Picamera2 class for capturing images
    picam2 = Picamera2()

    # Configure the preview settings for the camera (size, format, alignment)
    picam2.preview_configuration.main.size = (50, 50)
    picam2.preview_configuration.main.format = "BGR888"
    picam2.preview_configuration.align()
    picam2.configure("preview")

    # Start capturing images
    picam2.start()

    # Specify the number of images to capture for training
    images_to_capture = 100
    images_captured = 0

    try:
        # Continue capturing images until the desired number is reached
        while images_captured < images_to_capture:
            # Capture a single frame from the camera
            frame = picam2.capture_array()

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Iterate over detected faces
            for (x, y, w, h) in faces:
                # Extract the face region from the grayscale frame
                face_region = gray[y:y+h, x:x+w]

                # Resize the face region to a standard size (120x120)
                face_region_resized = cv2.resize(face_region, (120, 120))

                # Display the captured face region
                cv2.imshow('Captured Face', face_region_resized)

                # Append the resized face region to the images list
                images.append(face_region_resized)

                # Append the label ID to the labels list
                labels.append(label_id)

                # Increment the count of captured images
                images_captured += 1

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Print the number of images captured
            print('images captured : ' + str(images_captured))

    finally:
        # Close OpenCV windows and stop the camera when the loop is exited
        cv2.destroyAllWindows()
        picam2.stop()
        picam2.close()

    # Convert lists to NumPy arrays
    (images, labels) = [np.array(lis) for lis in [images, labels]]

    # Train the face recognition model with the captured images and labels
    model.train(images, labels)

    # Save the trained model to an XML file
    model.save(output_xml_file)


def recognize_faces(trained_model_file, frame):
    # Load the trained face recognition model
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(trained_model_file)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Set up the face detector
    face_cascade = cv2.CascadeClassifier('../hash.xml')

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Crop the face region
        face_region = gray[y:y+h, x:x+w]

        # Resize the face region to a standard size
        face_region_resized = cv2.resize(face_region, (112, 92))

        # Recognize the face using the trained model
        label, confidence = model.predict(face_region_resized)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the recognized name and confidence level
        recognized_name = f"Person {label} ({confidence:.2f})"
        cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame


print("1 = prendre photo")
print("2 = tester model")
while True:
    choix = int(input("Choisissez une option (1, 2, etc.): "))

    if choix == 1:
        infirmiereNom = input("Donnez le nom de l'infirmière : ")
        output_xml_file = infirmiereNom + '_model.xml'
        photoEntrainement(output_xml_file)
        break

    elif choix == 2:
        infirmiereNom = input("Donnez le nom de l'infirmière : ")
        trained_model_file = infirmiereNom + '_model.xml'

        picam2 = Picamera2()
        picam2.preview_configuration.main.size = (1280, 720)
        picam2.preview_configuration.main.format = "RGB888"
        picam2.preview_configuration.align()
        picam2.configure("preview")
        picam2.start()

        try:
            while True:
                # Capture the camera image
                im = picam2.capture_array()

                # Perform face recognition on the captured frame
                im_with_recognition = recognize_faces(trained_model_file, im)

                # Display the combined image with both raw camera and face recognition
                cv2.imshow("Camera with Face Recognition", im_with_recognition)

                # Break the loop when 'q' is pressed
                if cv2.waitKey(1) == ord('q'):
                    break

        finally:
            # Release resources
            cv2.destroyAllWindows()
            picam2.stop()
            picam2.close()
            break

    else:
        print('Option non reconnue. Essayez à nouveau.')
