# Imports
import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

# Constants
size = 4
classes = ['with_mask', 'without_mask']

model = load_model('keras_model.h5')

# Start Capturing
webcam = cv2.VideoCapture(0)

# Haar Cascade Classifier
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:

    ret, img = webcam.read()

    # Resize the image
    mini = cv2.resize(img, (img.shape[1] // size, img.shape[0] // size))

    # Detect faces
    faces = classifier.detectMultiScale(mini)

    for f in faces:
        (x, y, w, h) = [v * size for v in f] # Scale the image back again
        # Get the face
        face_img = img[y: y+h, x: x+w]

        # Save the face
        cv2.imwrite('temp.jpg', face_img)

        # Process the image
        test_img = load_img('temp.jpg', target_size=(256, 256, 3))
        test_img = img_to_array(test_img)
        test_img = test_img.reshape(-1, 256, 256, 3)  # Reshape the 0 axis
        test_img = (test_img.astype(np.float32) / 127.0) - 1  # Normalize 
        result = model.predict(test_img)
        
        # Predictions
        pred= np.argmax(result)
        print(pred)
        label = classes[pred]

        # Show predictions and bounding boxes
        cv2.rectangle(img,(x,y),(x+w,y+h), (0, 255, 0), 2)
        cv2.rectangle(img,(x,y-40),(x+w,y), (0, 255, 0), -1)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    # Show the image
    cv2.imshow('LIVE', img)
    key = cv2.waitKey(10)
    
    # if Esc key is press then break out of the loop 
    if key == 27: # The Esc key
        break

# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()


