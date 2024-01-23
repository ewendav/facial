import cv2
import os
import numpy

def ReconnaissanceFacial(name): #return true if the face correspond with the name, else false
    size = 2
    fn_haar = '../hash.xml'
    fn_dir = 'Entrainement'
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
    (images, lables) = [numpy.array(lis) for lis in [images, lables]]

    # OpenCV trains a model from the images
    # NOTE FOR OpenCV2: remove '.face'
    #model = cv2.face.createFisherFaceRecognizer()
    #model = cv2.face.FisherFaceRecognizer_create()
    model = cv2.face.LBPHFaceRecognizer_create()
    #model = cv2.face.EigenFaceRecognizer_create()
    model.train(images, lables)

    haar_cascade = cv2.CascadeClassifier(fn_haar)
    cam = cv2.VideoCapture(0)
    pasReconnu=True
    retour=False
    compteur=0
    while pasReconnu:
        ret, frame = cam.read()

        # Flip the image (optional)
        frame=cv2.flip(frame,1,0)

        # Convert to grayscalel
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize to speed up detection (optinal, change size above)
        mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

        # Detect faces and loop through each one
        faces = haar_cascade.detectMultiScale(mini)
        for i in range(len(faces)):
            face_i = faces[i]

            # Coordinates of face after scaling back by `size`
            (x, y, w, h) = [v * size for v in face_i]
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (im_width, im_height))

            prediction = model.predict(face_resize)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Try to recognize the face

            # [1]
            # Write the name of recognized face
            #cv2.putText(frame,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            if prediction[1]<90:
                cv2.putText(frame,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                #print(prediction[1])
                #print('%s - %.0f' % (names[prediction[0]],prediction[1]))
                if names[prediction[0]]==name:
                    retour=True
                    pasReconnu=False
            #cv2.putText(frame,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 0, 255))
            # Show the image and check for ESC being pressed
        cv2.imshow('OpenCV', frame)
    return retour

ReconnaissanceFacial('alexis')