import cv2
import os
from recognizer import train_recognizer, load_recognizer
from face_detection import detect_faces
from gui import create_gui, update_frame
from utils import open_camera, load_image

MODEL_FILE = "face_recognition_model.yml"
IMAGE_TEST_PATH = "images/test_image.jpg"
CONFIDENCE_THRESHOLD = 50
names = {1: "Persona1", 2: "Persona2", 3: "Persona3"}

# Entrenar el modelo si no existe
if not os.path.exists(MODEL_FILE):
    train_recognizer()

recognizer = load_recognizer()

# Intentamos abrir la cámara o usamos imagen
cap = None
try:
    cap = open_camera()
except Exception as e:
    print(f"Error al abrir la cámara: {e}")
    cap = None

if cap is None:
    test_image = load_image(IMAGE_TEST_PATH)

# Crear GUI
root, canvas = create_gui()

while True:
    if cap:
        ret, frame = cap.read()
        if not ret:
            break
    else:
        frame = test_image

    faces = detect_faces(frame)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        id_, confidence = recognizer.predict(face)

        name = names.get(id_, "Desconocido") if confidence < CONFIDENCE_THRESHOLD else "Desconocido"
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    update_frame(canvas, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar recursos y cerrar ventanas
if cap:
    cap.release()
cv2.destroyAllWindows()

root.mainloop()
