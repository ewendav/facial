
import cv2
import numpy as np
from picamera2 import Picamera2, Preview
import os
import sys
import time

class Camera : 

    def prendsPhotos(self):
        size = 4
        fn_haar = '../assets/hash.xml'
        fn_dir = '../assets/photos'
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

            cv2.namedWindow('facial recognition', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('facial recognition', 400, 300)  
            cv2.imshow('facial recognition', frame)
            key = cv2.waitKey(1) & 0xFF

            # If the 'Esc' key is pressed, break from the loop
            if key == 27:
                break

        # Release the camera resources
        camera.close()
        cv2.destroyAllWindows()



    def entrainementPhoto(self):
        
        size = 4
        fn_haar = '../assets/hash.xml'
        fn_dir = '../assets/photos'

        (images, lables, names, id) = ([], [], {}, 0)
        for (subdirs, dirs, files) in os.walk(fn_dir):
            for subdir in dirs:
                names[id] = subdir
                subjectpath = os.path.join(fn_dir, subdir)
                for filename in os.listdir(subjectpath):
                    f_name, f_extension = os.path.splitext(filename)
                    if(f_extension.lower() not in ['.png','.jpg','.jpeg','.gif','.pgm']):
                        continue
                    path = subjectpath + '/' + filename
                    lable = id
                    images.append(cv2.imread(path, 0))
                    lables.append(int(lable))
                id += 1
        (im_width, im_height) = (112, 92)
        print(images)
        # Create a Numpy array from the two lists above
        (images, lables) = [np.array(lis) for lis in [images, lables]]

        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(images, lables)

        model.save("../assets/models/" + 'master' + "_model.xml")
        print("Training complete. Model overrided as " + 'master' + "_model.xml")



    def ReconnaissanceFacial(self, name = ''):
        size = 4
        fn_haar = '../assets/hash.xml'
        folder_path = '../assets/photos'
        names = {}

        subdirectories = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        for label, subdir in enumerate(subdirectories):
            names[label] = subdir
        

        model = cv2.face.LBPHFaceRecognizer_create()
        model.read('../assets/models/' + 'master' + '_model.xml')  

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

            height, width, channels = frame.shape
            frame = cv2.flip(frame, 1, 0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

            # # Detect faces 
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
                    cv2.putText(frame,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,4,(0, 255, 0),thickness=4)
                    if name != '':
                        if names[prediction[0]] == name:
                            retour = True
                            pasReconnu = False
                            print(f"Face recognized: {name}")
                    
            cv2.namedWindow('facial recognition', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('facial recognition', 500, 500)  
            cv2.imshow('facial recognition', frame)

            key = cv2.waitKey(1)
            if key != -1:
                key = key & 0xFF
            
            if key == 27:
                break

        camera.close()
        cv2.destroyAllWindows()

        return retour

