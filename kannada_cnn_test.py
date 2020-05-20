import tensorflow as tf
import numpy as np
import cv2

# Load the model
new_model = tf.keras.models.load_model('kannada_number_reader.model')

# new_model.summary()

# Turn on web camera

cap = cv2.VideoCapture(0)

# Draw and check Predictions

run = False
ix, iy = -1, -1
follow = 25
img = np.zeros((512, 512, 1))


def draw(event, x, y, flag, params):
    global run, ix, iy, img, follow
    if event == cv2.EVENT_LBUTTONDOWN:
        run = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if run:
            cv2.circle(img, (x, y), 20, (255, 255, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        run = False
        cv2.circle(img, (x, y), 20, (255, 255, 255), -1)
        gray = cv2.resize(img, (28, 28))
        gray = gray.reshape(-1, 28, 28, 1)
        result = np.argmax(new_model.predict(gray))
        result = f'cnn: {result}'
        cv2.putText(img,
                    org=(25, follow),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    text=result,
                    color=(255, 0, 0),
                    thickness=1)
        follow += 20

        # Right click to clear the screen
    elif event == cv2.EVENT_RBUTTONDOWN:
        img = np.zeros((512, 512, 1))
        follow = 25


cv2.namedWindow('image')
cv2.setMouseCallback('image', draw)

while True:

    cv2.putText(img, 'Draw a Kannada number', (50, 480), cv2.FONT_HERSHEY_DUPLEX, 1,
                (255, 0, 0), 1)
    cv2.imshow("image", img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
