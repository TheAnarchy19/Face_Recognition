import cv2

def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("No se puede acceder a la cámara.")
    return cap

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error al cargar la imagen: {image_path}")
    return image
