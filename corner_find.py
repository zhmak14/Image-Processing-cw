import cv2

def get_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"point: ({x}, {y})")

img = cv2.imread("driving_images\im005-snow.jpg")
cv2.namedWindow("click corners", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("click corners", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("click corners", img)
cv2.setMouseCallback("click corners", get_mouse_click)
cv2.waitKey(0)
cv2.destroyAllWindows()