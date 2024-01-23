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
    (images, labels, label_id) = ([], [], 0)

    model = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier('../hash.xml')

    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (200, 200)
    picam2.preview_configuration.main.format = "BGR888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    images_to_capture = 30
    images_captured = 0

    try:
        while images_captured < images_to_capture:
            frame = picam2.capture_array()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_region = gray[y:y+h, x:x+w]

                face_region_resized = cv2.resize(face_region, (120, 120))
                cv2.imshow('Captured Face', face_region_resized)
                images.append(face_region_resized)
                labels.append(label_id)
                images_captured += 1

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print('images captured : ' + str(images_captured) )


    finally:
        cv2.destroyAllWindows()
        picam2.stop()
        picam2.close()

    (images, labels) = [np.array(lis) for lis in [images, labels]]

    model.train(images, labels)

    model.save(output_xml_file)



infirmiereNom = input("nom de l'infirmière")

output_xml_file = infirmiereNom + '_model.xml'
photoEntrainement(output_xml_file)


# import cv2
# from picamera2 import Picamera2
# import numpy as np

# def recognize_faces(trained_model_file, frame):
#     # Load the trained face recognition model
#     model = cv2.face.LBPHFaceRecognizer_create()
#     model.read(trained_model_file)

#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Set up the face detector
#     face_cascade = cv2.CascadeClassifier('../hash.xml')

#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     for (x, y, w, h) in faces:
#         # Crop the face region
#         face_region = gray[y:y+h, x:x+w]

#         # Resize the face region to a standard size
#         face_region_resized = cv2.resize(face_region, (112, 92))

#         # Recognize the face using the trained model
#         label, confidence = model.predict(face_region_resized)

#         # Draw a rectangle around the face
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#         # Display the recognized name and confidence level
#         recognized_name = f"Person {label} ({confidence:.2f})"
#         cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#     return frame

# def main():
#     trained_model_file = 'nino_model.xml'

#     picam2 = Picamera2()
#     picam2.preview_configuration.main.size = (1280, 720)
#     picam2.preview_configuration.main.format = "RGB888"
#     picam2.preview_configuration.align()
#     picam2.configure("preview")
#     picam2.start()

#     try:
#         while True:
#             # Capture the camera image
#             im = picam2.capture_array()

#             # Perform face recognition on the captured frame
#             im_with_recognition = recognize_faces(trained_model_file, im)

#             # Display the combined image with both raw camera and face recognition
#             cv2.imshow("Camera with Face Recognition", im_with_recognition)

#             # Break the loop when 'q' is pressed
#             if cv2.waitKey(1) == ord('q'):
#                 break

#     finally:
#         # Release resources
#         cv2.destroyAllWindows()
#         picam2.stop()
#         picam2.close()

# if __name__ == "__main__":
#     main()
