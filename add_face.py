import cv2
import numpy as np
import face_recognition

def main(name):
    cap = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
    while(cap.isOpened()):
        success, photo = cap.read()
        faces = faceCascade.detectMultiScale(
            cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY),
            scaleFactor = 1.5,
            minNeighbors = 6, 
            minSize = (30, 30)
        )
        for (x, y, w, h) in faces:
            if cv2.waitKey(25) & 0xFF == ord('s'):
                cv2.imwrite('face/'+ str(name) + '.jpg', mirrorReflection(photo))
                cap.release()
                return
            cv2.rectangle(photo, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('VideoCapture', mirrorReflection(photo))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()

def fullFace(x, y, w, h, photo, SIZE):
    cropped = photo[y:y + w, x:h + x]
    r = float(SIZE) / cropped.shape[1]
    dim = (SIZE, int(cropped.shape[0] * r))
    return cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA)

def mirrorReflection(photo):
    return np.flip(photo, axis = 1)

if __name__ == '__main__':
    NAME = input('_NAME_FACE_ :')
    main(NAME)
    cv2.destroyAllWindows()

    # image = face_recognition.load_image_file("face/" + str(NAME) + ".jpg")
    # face_encoding = face_recognition.face_encodings(image)[0]

    # with open("face_recognition/" + NAME + '.face_recognition', 'w') as f:
    #     for row in face_encoding:
    #         f.write(str(row))
    #         f.write('\n')
    # f.close()