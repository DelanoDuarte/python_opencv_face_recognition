#from imageai.Detection import VideoObjectDetection
import cv2

camera = cv2.VideoCapture(0)
track_window = (250, 90, 400, 125)

head_cascade_classfier = cv2.CascadeClassifier(
    'C:/Users/Delano Jr/Documents/Desenvolvimento/Desenvolvimento Python/bots/haarcascade_frontalface_default.xml')

while True:
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = head_cascade_classfier.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('img', img)

    #cv2.imshow('frame', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
camera.release()

# video_dectection = VideoObjectDetection()

# video_dectection.setModelTypeAsYOLOv3()
# video_dectection.setModelPath('')
# video_dectection.loadModel()
