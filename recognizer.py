import cv2
import os
import numpy as np

BASE_DIR = "faces_database"
MODEL_FILE = "face_recognition_model.yml"

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

def load_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)
    return recognizer
