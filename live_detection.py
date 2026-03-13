import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import serial
import time

# ---------------- CONFIG ----------------
MODEL_PATH = "models/garbage_model.pth"
IMG_SIZE = 224
ARDUINO_PORT = "COM3"   # CHANGE if needed
BAUD_RATE = 9600

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------- LOAD MODEL -------------
checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint["classes"]

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features, len(class_names)
)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

print("Loaded classes:", class_names)

# ------------- HUMAN (FACE) DETECTION ----------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ------------- IMAGE TRANSFORM ----------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------- ARDUINO -------------------
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("Arduino connected")
except:
    arduino = None
    print("Arduino NOT connected")

# ------------- CAMERA -------------------
cap = cv2.VideoCapture(0)
print("Live detection started. Press ESC to exit.")

last_sent = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # -------- HUMAN DETECTION ----------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    human_detected = len(faces) > 0

    # -------- ROI BOX ----------
    box_size = 400
    x1 = w // 2 - box_size // 2
    y1 = h // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    roi = frame[y1:y2, x1:x2]
    roi = cv2.convertScaleAbs(roi, alpha=1.0, beta=-30)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # -------- PREDICTION ----------
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        probs = F.softmax(out, dim=1)[0]

    prob_dict = {
        class_names[i]: probs[i].item() * 100
        for i in range(len(class_names))
    }

    if "Wet" in prob_dict:
        prob_dict["Wet"] *= 1.25
    if "Dry" in prob_dict:
        prob_dict["Dry"] *= 0.85

    label = max(prob_dict, key=prob_dict.get)
    confidence = min(prob_dict[label], 100.0)

    # -------- DISPLAY ----------
    cv2.putText(
        frame,
        f"{label} : {confidence:.1f}%",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # -------- HUMAN SAFETY ----------
    if human_detected:
        cv2.putText(
            frame,
            "HUMAN DETECTED - BIN LOCKED",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )
    else:
        # -------- SEND TO ARDUINO ----------
        if arduino:
            if label != last_sent:
                if label == "Dry":
                    arduino.write(b"D\n")
                elif label == "Wet":
                    arduino.write(b"W\n")
                elif label == "Metal":
                    arduino.write(b"M\n")
                last_sent = label

    cv2.imshow("Garbage Live Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
