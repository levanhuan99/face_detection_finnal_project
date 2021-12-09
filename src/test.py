from cv2 import cv2

key = cv2.waitKey(10000)
if key % 256 == 27:
    # ESC pressed
    print("Escape hit, closing...")
