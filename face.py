import cv2
from google.colab.patches import cv2_imshow

face_cascade = cv2.CascadeClassifier('/content/haarcascade_frontalface_default.xml')
img = cv2.imread("/content/IMG_20210704_142217.jpg")
cv2_imshow(img)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2_imshow(gray)

faces = face_cascade.detectMultiScale(gray)
print(faces)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 100, 0), 2)
cv2_imshow(img)

eye_cascade = cv2.CascadeClassifier('/content/haarcascade_eye.xml')
img = cv2.imread("/content/IMG_20210704_142217.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
eyes = eye_cascade.detectMultiScale(gray)
print(eyes)
for (x, y, w, h) in eyes:
    cv2.rectangle(img, (x, y), (x+w, y+h), (100, 10, 250), 2)
cv2_imshow(img)

//using function
def fun_to_detect(img_path):
    face_cascade = cv2.CascadeClassifier('/content/haarcascade_frontalface_default.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    print(faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (100, 10, 250), 2)
    cv2_imshow(img)
fun_to_detect("/content/IMG-20210627-WA0032.jpg")
