```python
import cv2
import numpy as np


def preprocess_image(image_path):
   
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, img_thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        raise ValueError("No contours found in the image.")


    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    digit = img_thresh[y:y+h, x:x+w]
    if w > h:
        new_w = 20
        new_h = int((h / w) * 20)
    else:
        new_h = 20
        new_w = int((w / h) * 20)

    digit_resized = cv2.resize(digit, (new_w, new_h))

    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized

    img_normalized = canvas / 255.0

    img_flatten = img_normalized.flatten().reshape(1, -1).astype(np.float32)

    return img_flatten



```
