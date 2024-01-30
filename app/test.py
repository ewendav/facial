# import cv2
# from picamera2 import Picamera2

# # Load the pre-trained Haar Cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier('../hash.xml')

# camera = Picamera2()
# camera.preview_configuration.main.size = (1280, 720)
# camera.preview_configuration.main.format = "RGB888"
# camera.preview_configuration.align()
# camera.configure("preview")
# camera.start()

# try:
#     while True:
#         # Capture the camera image
#         im = camera.capture_array()

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
#     camera.stop()
#     camera.close()



import cv2
import numpy as np
from picamera2 import Picamera2, Preview
import os
import cv2
import sys
import time

# sys.path.append('dependencies/picameraFolder/picamera')  


def photoEntrainement():
    size = 4
    fn_haar = '../hash.xml'
    fn_dir = 'Photos'
    count_max = 30

    try:
        print("Identifiant de l'infirmière")
        fn_name = input()
        if len(fn_name) == 0:
            print("Vous devez fournir un identifiant valide !")
            sys.exit(0)
        print("Nom du dossier des Photos, enter si 'Photos'")
        fn_dir1 = input()
        if len(fn_dir1) > 0:
            fn_dir = fn_dir1
        print("Nombre de photos, par défaut 30 enter si ok")
        count_max1 = input()
        if len(count_max1) > 0:
            count_max = int(str(count_max1))
    except:
        print("Erreur de saisie !")
        sys.exit(0)

    path = os.path.join(fn_dir, fn_name)
    if not os.path.isdir(path):
        os.mkdir(path)

    (im_width, im_height) = (112, 92)
    haar_cascade = cv2.CascadeClassifier(fn_haar)

    # Create the PiCamera2 object
    camera = Picamera2()
    camera.resolution = (640, 480)

    pin = sorted([int(n[:n.find('.')]) for n in os.listdir(path) if n[0] != '.'] + [0])[-1] + 1

    print("\n\033[94mLe programme va enregistrer " + str(count_max) + " photos. \
    Veuillez bouger la tête pour prendre des photos de face différenciées.\033[0m\n")
    camera_config = camera.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
    camera.configure(camera_config)

    count = 0
    pause = 0
    camera.start()

    while True:
        frame = None
        try:
            # Capture a frame
            # camera.start_preview(Preview.QTGL)

            # camera.start()
            time.sleep(1)
            photo = camera.capture_array("main")


        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error capturing frame: {e}")
            continue

        # Get the NumPy array representing the image
        frame = photo

        # Get image size
        height, width, channels = frame.shape

        # Flip frame
        frame = cv2.flip(frame, 1, 0)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Scale down for speed
        mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

        # Detect faces
        faces = haar_cascade.detectMultiScale(mini)

        # We only consider the largest face
        faces = sorted(faces, key=lambda x: x[3])
        if faces:
            face_i = faces[0]
            (x, y, w, h) = [v * size for v in face_i]

            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (im_width, im_height))

            # Draw a rectangle and write the identifier
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, fn_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

            # Do not consider faces that are too small
            if (w * 6 < width or h * 6 < height):
                print("Face trop petite")
            else:
                # To create diversity, only save every fifth detected image
                if (pause == 0):
                    print("Sauvegarde de la photo " + str(count + 1) + "/" + str(count_max))

                    # Save image file
                    cv2.imwrite('%s/%s.png' % (path, pin), face_resize)

                    pin += 1
                    count += 1
                    pause = 1

        if (pause > 0):
            pause = (pause + 1) % 5

        cv2.imshow('OpenCV', frame)
        key = cv2.waitKey(1) & 0xFF

        # If the 'Esc' key is pressed, break from the loop
        if key == 27:
            break

    # Release the camera resources
    camera.close()
    cv2.destroyAllWindows()






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

while True:
    print("1 = prendre photo")
    print("2 = tester model")

    choix = int(input("Choisissez une option (1, 2, etc.): "))

    if choix == 1:
        photoEntrainement()
        break

    elif choix == 2:
        infirmiereNom = input("Donnez le nom de l'infirmière : ")
        trained_model_file = infirmiereNom + '_model.xml'

        camera = Picamera2()
        camera.preview_configuration.main.size = (1280, 720)
        camera.preview_configuration.main.format = "RGB888"
        camera.preview_configuration.align()
        camera.configure("preview")
        camera.start()

        try:
            while True:
                # Capture the camera image
                im = camera.capture_array()

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
            camera.stop()
            camera.close()
            break

    else:
        print('Option non reconnue. Essayez à nouveau.')
