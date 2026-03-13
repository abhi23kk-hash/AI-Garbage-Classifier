import cv2
import os

# CHANGE THIS to Dry / Wet / Metal before running
CATEGORY = "Metal"
SAVE_DIR = f"data/train/{CATEGORY}"

os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Press SPACE to capture image")
print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Images", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        img_path = os.path.join(SAVE_DIR, f"{CATEGORY}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        print("Saved:", img_path)
        count += 1

cap.release()
cv2.destroyAllWindows()
