from Camera import Camera

camera = new Camera()

while True:
    print("1 = prendre photo")
    print('2 = entrainer le model sur les photos')
    print("3 = tester model")

    choix = int(input("Choisissez une option (1, 2, 3.): "))

    if choix == 1:
        camera.prendsPhotos()
        print('photos prises')
        print('entrainement du model sur toutes les dossier de Photos ')
        camera.entrainementPhoto()

    elif choix == 2:
        print('entrainement du model sur toutes les dossier de Photos ')
        camera.entrainementPhoto()

    elif choix == 3:
        camera.ReconnaissanceFacial()

    else:
        print('Option non reconnue. Essayez à nouveau.')
