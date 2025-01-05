# Proyecto sobre Reconocimiento Facial

Este proyecto utiliza OpenCV para realizar el reconocimiento facial en imágenes y video en tiempo real. El sistema está diseñado para detectar y reconocer rostros utilizando el clasificador de características de Haar y un modelo de reconocimiento facial LBPH (Local Binary Patterns Histograms).

## Características

- **Entrenamiento Automático**: El sistema entrena el modelo de reconocimiento facial a partir de imágenes almacenadas en una base de datos de rostros.
- **Detección de Rostros**: Detecta rostros en imágenes o video en tiempo real usando un clasificador de Haar.
- **Interfaz Gráfica**: Una interfaz gráfica básica con Tkinter para visualizar los resultados.
- **Soporte para Cámara y Archivos de Imágenes**: El sistema intenta usar la cámara del dispositivo, pero si no está disponible, utiliza una imagen estática de prueba.


## Instalación

### Requisitos

Asegúrate de tener las siguientes dependencias instaladas:

- `opencv-python`
- `opencv-python-headless`
- `numpy`
- `Pillow`
- `tkinter` (si no está instalado en tu entorno, puedes instalarlo con `sudo apt-get install python3-tk` en sistemas basados en Debian)

Instala las dependencias usando `pip`:

```bash
pip install opencv-python opencv-python-headless numpy Pillow
```

### Licencia
Este `README.md` incluye información clave sobre la instalación, ejecución y estructura del proyecto, además de explicar cómo funciona y cómo puedes contribuir.
