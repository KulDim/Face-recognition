import face_recognition
import cv2
import numpy as np
import os

from modulus.FPS import FPS; FPS = FPS(); SEVE_FPS = int()

video_capture = cv2.VideoCapture(0)



# Settings
INT_DELAY_TIME = 30
progressBar = 0
peopleSearch = True
delay = INT_DELAY_TIME
_is_open = bool


known_face_encodings = []
known_face_names = []


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
    if peopleSearch:
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








            if True in matches:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            if True in matches:
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            else:
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            if True in matches:
                progressBar = progressBar + 10

            else:
                progressBar = progressBar - 10

            if progressBar >= 110:
                progressBar = 0
                peopleSearch = False
                _is_open = True

            elif progressBar <= -60:
                progressBar = 0
                peopleSearch = False
                _is_open = False

    if not peopleSearch:
        delay = delay - 1
        font = cv2.FONT_HERSHEY_DUPLEX

        if _is_open:
            cv2.putText(frame, 'OPEN THE DOOR', (10, 30), font, 1.0, (0, 255, 0), 5)
        if not _is_open:
            cv2.putText(frame, 'GO AWAY', (10, 30), font, 1.0, (0, 0, 255), 5)

        if(delay <= 0):
            delay = INT_DELAY_TIME
            peopleSearch = True
        else:
            print('DELAY: ' + str(delay))

    print('progressBar: ' + str(progressBar))
    font = cv2.FONT_HERSHEY_DUPLEX

    # Frames per second
    counter = FPS.frameСounter()
    if counter != None: SEVE_FPS = counter

    cv2.putText(frame, 'FPS: ' + str(int(SEVE_FPS)), (10, 65), font, 1.0, (0, 0, 255), 5)

    cv2.putText(frame, 'progressBar :' + str(progressBar) + '%', (10, 100), font, 1.0, (0, 0, 255), 5)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


