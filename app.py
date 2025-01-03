import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

BASE_DIR = "faces_database"
MODEL_FILE = "face_recognition_model.yml"
IMAGE_TEST_PATH = "test_image.jpg"  # Cambia esta ruta a una imagen que quieras usar

def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    image_paths = [os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR)]
    face_samples = []
    ids = []

    for image_path in image_paths:
        try:
            filename = os.path.basename(image_path)
            if "_" not in filename or not filename.endswith(".jpg"):
                print(f"Archivo ignorado: {filename}")
                continue

            id_ = int(filename.split("_")[1].split(".")[0])

            gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

            if len(faces) == 0:
                print(f"No se detectaron rostros en la imagen {filename}")
            for (x, y, w, h) in faces:
                face_samples.append(gray_image[y:y + h, x:x + w])
                ids.append(id_)

        except Exception as e:
            print(f"Error procesando el archivo {image_path}: {e}")

    if not face_samples or not ids:
        raise ValueError("No se encontraron suficientes datos para entrenar. Verifica la base de datos de imágenes.")

    recognizer.train(face_samples, np.array(ids))
    recognizer.save(MODEL_FILE)
    print("Modelo entrenado y guardado correctamente.")

if not os.path.exists(MODEL_FILE):
    train_recognizer()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_FILE)

# Diccionario de nombres de personas
names = {1: "Persona1", 2: "Persona2", 3: "Persona3"} 

# Cargamos el clasificador en escala de grises
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

CONFIDENCE_THRESHOLD = 50

# Intentamos abrir la cámara, si falla usamos una imagen
cap = None
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("No se puede acceder a la cámara. Usando imágenes en su lugar.")
    print("Cámara abierta correctamente.")
except Exception as e:
    print(f"Error al abrir la cámara: {e}")
    cap = None

if cap is None:
    print("Usando imagen de prueba en lugar de la cámara.")
    test_image = cv2.imread(IMAGE_TEST_PATH)
    if test_image is None:
        print(f"Error: No se pudo cargar la imagen en {IMAGE_TEST_PATH}.")
        exit()

# Configuración de Tkinter
root = tk.Tk()
root.title("Reconocimiento Facial")

# Crear un canvas para mostrar el video
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

def update_frame(frame):
    """ Actualiza la imagen en el canvas con el frame procesado """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.image = img_tk  # Para evitar que se elimine la referencia de la imagen

while True:
    if cap:
        ret, frame = cap.read()
        if not ret:
            break
    else:
        frame = test_image

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Ajustamos los parámetros de la detección de rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        id_, confidence = recognizer.predict(face)

        if confidence < CONFIDENCE_THRESHOLD:
            name = names.get(id_, "Desconocido")
        else:
            name = "Desconocido" 

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    update_frame(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar recursos y cerrar ventanas
if cap:
    cap.release()
cv2.destroyAllWindows()

root.mainloop()
