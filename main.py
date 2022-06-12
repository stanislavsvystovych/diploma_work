import os
import cv2
import numpy as np
from keras import models

capture = cv2.VideoCapture("sample_videos\\3.mp4")
# capture = cv2.VideoCapture(0)

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
cascade_path = os.path.join(cv2_base_dir, "data\\haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

gender_recognition_model = models.load_model("models\\gender_recognition.h5")
age_recognition_model = models.load_model("models\\age_recognition.h5")

while capture.isOpened():
    ret, frame = capture.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_coordinates = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3,
                                                      minNeighbors=5, minSize=(40, 40))

    faces = []
    for (i, (x, y, w, h)) in enumerate(faces_coordinates):
        cropped_image = frame[y:y + h, x:x + w]
        resized_image = cv2.resize(cropped_image, (200, 200))
        cv2.imshow("Face", resized_image)
        normalized_image = resized_image / 255.0
        faces.append(normalized_image)

    if len(faces) != 0:
        gender_predictions = gender_recognition_model.predict(np.array(faces), verbose=0)
        age_predictions = age_recognition_model.predict(np.array(faces), verbose=0)

        for i in range(len(faces)):
            gender = ""
            if gender_predictions[i][0] < 0.5:
                gender = "Male"
            else:
                gender = "Female"

            age = round(age_predictions[i][0] * 116)

            (x, y, w, h) = faces_coordinates[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, gender + ", " + str(age), (x, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.namedWindow("Diploma Work")
    cv2.imshow("Diploma Work", frame)

    if cv2.waitKey(1) == ord("q"):
        break

# When everything is done, release the capture
capture.release()
cv2.destroyAllWindows()