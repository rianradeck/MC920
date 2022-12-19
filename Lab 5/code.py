import cv2 as cv
import numpy as np

try:
    n = int(input("1 - dvd\n2 - Kael\n3 - Rian\n4 - Carros\n5 - Câmera\n(Pressione Q para sair durante a execução)\n"))
except:
    exit()

if n == 1:
    cap = cv.VideoCapture("bouncing dvd logo.m4a") # simple example
if n == 2:
    cap = cv.VideoCapture("video.mp4")
if n == 3:
    cap = cv.VideoCapture("video2.mp4") 
if n == 4:
    cap = cv.VideoCapture("Cars Tuner Scene _ Pixar Cars.m4a") # complex example
if n == 5:
    cap = cv.VideoCapture(0) # camera

last = np.array([None])

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if last.any() == None:
            last = gray
            continue

        diff = np.abs(gray.astype("int16") - last.astype("int16")).astype("uint8")
        img = cv.GaussianBlur(diff, (11, 11), 5)
        ret, img = cv.threshold(img, 30, 255, cv.THRESH_BINARY)
        img = cv.dilate(img, (100, 100))

        dilatation_size = 30
        dilation_shape = cv.MORPH_ELLIPSE
        element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1), (dilatation_size, dilatation_size))
        img = cv.dilate(img, element)

        contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv.contourArea(contour) < 1000:
                continue

            rect = cv.boundingRect(contour)
            frame = cv.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 255, 0))
        
        cv.imshow('Frame', frame)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else: 
        break

    last = gray
 
cap.release()

cv.destroyAllWindows()