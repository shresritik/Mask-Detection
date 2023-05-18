import cv2

import tensorflow as tf
face = cv2.CascadeClassifier(
    "D:/python/image processing/face_detection/haarcascades/haarcascade_frontalface_default.xml")

model = tf.keras.models.load_model('face_mask1.h5')


def detect_face_mask(img):
    y_pred = model.predict(img.reshape(1, 224, 224, 3))
    return ((y_pred[0][0]))


# sam1 = cv2.imread('1.png')
# sam1 = cv2.resize(sam1, (224, 224))
# detect_face_mask(sam1)


def draw_label(img, text, pos, bg_color):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, cv2.FILLED)
    end_x = pos[0]+text_size[0][0]+2
    end_y = pos[1]+text_size[0][1]-2
    cv2.rectangle(img, pos, (end_x, end_y), bg_color, cv2.FILLED)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 0, 0), 1, cv2.LINE_AA)


cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    img = cv2.resize(frame, (224, 224))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    y_pred = detect_face_mask(img)

    if y_pred > 0.5:
        print("No mask", y_pred)
        draw_label(frame, "No Mask", (30, 30), (0, 0, 255))
    else:
        print("Mask", y_pred)
        draw_label(frame, "Mask", (30, 30), (0, 255, 0))
        # detect faces
    faces = face.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.imshow("window", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
