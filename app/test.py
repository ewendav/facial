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


def prendsPhotos():
    size = 4
    fn_haar = '../hash.xml'
    fn_dir = 'Photos'
    count_max = 30

    # try:
    #     print("Identifiant de l'infirmière")
    #     fn_name = input()
    #     if len(fn_name) == 0:
    #         print("Vous devez fournir un identifiant valide !")
    #         sys.exit(0)
    #     print("Nom du dossier des Photos, enter si 'Photos'")
    #     fn_dir1 = input()
    #     if len(fn_dir1) > 0:
    #         fn_dir = fn_dir1
    #     print("Nombre de photos, par défaut 30 enter si ok")
    #     count_max1 = input()
    #     if len(count_max1) > 0:
    #         count_max = int(str(count_max1))
    # except:
    #     print("Erreur de saisie !")
    #     sys.exit(0)

    # path = os.path.join(fn_dir, fn_name)
    # if not os.path.isdir(path):
    #     os.mkdir(path)

    (im_width, im_height) = (112, 92)
    haar_cascade = cv2.CascadeClassifier(fn_haar)

    # Create the PiCamera2 object
    camera = Picamera2()
    camera_config = camera.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
    camera.configure(camera_config)

    pin = sorted([int(n[:n.find('.')]) for n in os.listdir(path) if n[0] != '.'] + [0])[-1] + 1

    print("\n\033[94mLe programme va enregistrer " + str(count_max) + " photos. \
    Veuillez bouger la tête pour prendre des photos de face différenciées.\033[0m\n")


    count = 0
    pause = 0
    camera.start()
    time.sleep(2)


    while count < 30:
        frame = None
        try:
            frame = camera.capture_array("main")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error capturing frame: {e}")
            continue


        height, width, channels = frame.shape
        frame = cv2.flip(frame, 1, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

    return fn_name




def entrainementPhoto(data_folder, name):
    # Initialize face recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Get the paths of all images in the data folder
    image_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.jpg') or f.endswith('.png')]

    # Create lists to store face samples and corresponding labels
    face_samples = []
    labels = []

    # Read images and collect face samples and labels
    for image_path in image_paths:
        # Read the image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Extract face and label from image path
        label = int(os.path.split(image_path)[-1].split(".")[0])
        face_samples.append(img)
        labels.append(label)

    # Convert lists to NumPy arrays
    face_samples = np.asarray(face_samples, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.int32)

    # Train the face recognizer
    face_recognizer.train(face_samples, labels)

    # Save the trained model
    face_recognizer.save("models/" + name + "_model.xml")

    print("Training complete. Model saved as " + name + "_model.xml")



def ReconnaissanceFacial(name):
    size = 4
    fn_haar = '../hash.xml'
    names = {}

    # Load the pre-trained model
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read('models/' + name + '_model.xml')  

    (im_width, im_height) = (112, 92)
    haar_cascade = cv2.CascadeClassifier(fn_haar)

    camera = Picamera2()
    camera_config = camera.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
    camera.configure(camera_config)
    camera.start()
    time.sleep(2)

    pasReconnu = True
    retour = False
    
    while pasReconnu:
        frame = None

        try:
            frame = camera.capture_array("main")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error capturing frame: {e}")
            continue

        cv2.imshow('OpenCV', frame)


        height, width, channels = frame.shape
        frame = cv2.flip(frame, 1, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

        # Detect faces 
        faces = haar_cascade.detectMultiScale(mini)

        for i in range(len(faces)):
            face_i = faces[i]

            # Coordinates of face after scaling back by `size`
            (x, y, w, h) = [v * size for v in face_i]
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (112, 92))  

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            prediction = model.predict(face_resize)

            # Try to recognize the face
            if prediction[1] < 90:
                cv2.putText(frame, prediction[1], (x - 10, y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                if names[prediction[0]] == name:
                    retour = True
                    pasReconnu = False
                    print(f"Face recognized: {name}")




    camera.close()
    cv2.destroyAllWindows()

    return retour






while True:
    print("1 = prendre photo")
    print("2 = tester model")

    choix = int(input("Choisissez une option (1, 2, etc.): "))

    if choix == 1:
        name = prendsPhotos()
        print('photos prises, entrainement du model en cours')

        cheminPhotos = 'Photos/' + name

        entrainementPhoto(cheminPhotos, name)

        break

    elif choix == 2:
        infirmiereNom = input("Donnez le nom de l'infirmière : ")
        trained_model_file = infirmiereNom + '_model.xml'

        ReconnaissanceFacial(infirmiereNom)

    else:
        print('Option non reconnue. Essayez à nouveau.')
