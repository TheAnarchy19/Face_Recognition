import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

def create_gui():
    root = tk.Tk()
    root.title("Reconocimiento Facial")
    canvas = tk.Canvas(root, width=640, height=480)
    canvas.pack()
    return root, canvas

def update_frame(canvas, frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.image = img_tk  # Mantener la referencia a la imagen
