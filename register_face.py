import imutils
from imutils.video import VideoStream
import cv2
import os
import time

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# create a folder with the user's name
name = input('Enter your name: ')

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error creating new directory. ' + directory)

createFolder('./dataset/' + name + '/')

# start the video stream
vs = VideoStream(src=0).start()
time.sleep(2.0)

n = 0
key = 0
path = './dataset/' + name + '/'

while True:
    #while n <= 50:
    # Capture frame-by-frame
    frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        if n < 50:
            filename=str(n) + '.jpg'
            img = frame
            cv2.imwrite(os.path.join(path , filename), img)
            cv2.waitKey(3)
            n += 1
        elif n == 50:
            print('Face Registered')
            n += 1
    if n > 50:
        text = 'Face Registered'
        text2 = 'Please Press "Q"'
        cv2.putText(frame, text, (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
        cv2.putText(frame, text2, (30, 300),
            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Taking pictures', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()