import face_recognition
import cv2
import numpy as np
import os

from numpy.lib.shape_base import tile

from modulus.FPS import FPS; FPS = FPS(); SEVE_FPS = int()

video_capture = cv2.VideoCapture(0)

# Settings
DELAY = INT_DELAY_TIME = 30
IS_OPEN = bool
progressBar = 0
peopleSearch = False
peopleFase = True

font = cv2.FONT_HERSHEY_DUPLEX
known_face_encodings = []
known_face_names = []
faceCascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

def getPictures(directory):
    check = False
    files = os.listdir(directory)
    for img in files:
        (name, _) = img.split('.')
        face_encoding = face_recognition.face_encodings(face_recognition.load_image_file(directory + img))[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)
        check = True
    return check

openFileImg = getPictures('face/')

while openFileImg:
    ret, frame = video_capture.read()

    if  peopleFase:
        faces = faceCascade.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            scaleFactor = 1.5,
            minNeighbors = 6, 
            minSize = (30, 30)
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            peopleSearch = True
            peopleFase = False

    if  peopleSearch:
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            # Свой не свой 
            if True in matches:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                progressBar = (progressBar + 10)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                progressBar = (progressBar - 10)

            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            if progressBar >= 100 + 10: 
                progressBar = 0
                peopleSearch = False
                IS_OPEN = True
            elif progressBar <= -50 - 10:
                progressBar = 0
                peopleSearch = False
                IS_OPEN = False

    # DELAY OPEN OR NOT OPEN
    if not peopleFase:
        if not peopleSearch:
            DELAY = (DELAY - 1)
            if IS_OPEN: cv2.putText(frame, 'OPEN THE DOOR', (10, 30), font, 1.0, (0, 255, 0), 5)
            else: cv2.putText(frame, 'GO AWAY', (10, 30), font, 1.0, (0, 0, 255), 5)
            if(DELAY <= 0):
                DELAY = INT_DELAY_TIME
                peopleFase = True
                peopleSearch = False
            else: cv2.putText(frame, 'DELAY: ' + str(DELAY), (10, 140), font, 1.0, (0, 0, 255), 5)

    # Frames per second
    counter = FPS.frameСounter()
    if counter != None: SEVE_FPS = counter
    cv2.putText(frame, 'FPS: ' + str(int(SEVE_FPS)), (10, 65), font, 1.0, (0, 0, 255), 5)
    # progressBar
    cv2.putText(frame, 'progressBar :' + str(progressBar) + '%', (10, 100), font, 1.0, (0, 0, 255), 5)
    
    cv2.imshow('Video', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'): break

video_capture.release()
cv2.destroyAllWindows()