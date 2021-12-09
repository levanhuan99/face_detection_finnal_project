import numpy as np
import cv2
import pickle
import requests
URL = "http://192.168.0.104/open-door"
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

labels = {"person_name": 1}

username ="";
with open("pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)
same_percent = [""]
if cv2.waitKey(32):
    while (True):
        # Capture frame-by-frame

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            # print(x,y,w,h)
            roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
            roi_color = frame[y:y + h, x:x + w]

            # recognize? deep learned model predict keras tensorflow pytorch scikit learn
            id_, conf = recognizer.predict(roi_gray)
            if conf >= 40:
                # print(5: #id_)
                # print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
                same_percent.append(name)
                print("conf: " + str(conf) + "; name: " + str(name))
                username = str(name);
            img_item = "7.png"
            cv2.imwrite(img_item, roi_color)

            color = (255, 0, 0)  # BGR 0-255
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        # subitems = smile_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in subitems:
        #	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        # Display the resulting frame
        common_name = max(set(same_percent), key=same_percent.count)
        fred = 0
        finnal_result = 0
        if len(same_percent) == 30:

            for i in same_percent:
                if i == common_name:
                    fred += 1
            finnal_result = fred / len(same_percent)

            print("final result: "+str(finnal_result * 100))


            if finnal_result*100 >90:
                # TODO calling api open the door when finnal_result*100>90...
                URL += "?current_user_name="+username
                print("json date response: " + URL)
                r = requests.get(url=URL)
                data = r.json()
                cv2.destroyAllWindows()
        cv2.imshow('frame', frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
