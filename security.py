import face_recognition
import cv2
import numpy as np
import os

video_capture = cv2.VideoCapture(0)

known_face_encodings = []
known_face_names = []


is_file_img = False
directory = 'face/'
files = os.listdir(directory)
for img in files:
    (name, type) = img.split('.')
    face_encoding = face_recognition.face_encodings(face_recognition.load_image_file(directory + img))[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)
    is_file_img = True

while is_file_img:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

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
            print('open the door')

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


if(not is_file_img):
    print('Not face/file << RUN >> "add_face.py"')