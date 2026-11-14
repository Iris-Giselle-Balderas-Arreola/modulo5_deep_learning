"""Se implementó la predicción del tipo de escena de una fotografía."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
from pathlib import Path
import json
import argparse
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

BASE = Path(__file__).resolve().parent

def cargar_imagen(ruta, lado=128):
    """Se corrigió la orientación y se convirtió la imagen a RGB."""
    with Image.open(ruta) as imagen:
        imagen = ImageOps.exif_transpose(imagen).convert("RGB")
        imagen = imagen.resize((lado, lado), Image.Resampling.BILINEAR)
        return np.asarray(imagen, dtype="float32")

def predecir_imagen(ruta, modelo=None):
    """Se clasificó una fotografía nueva usando el modelo guardado."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró la imagen: {ruta}")
    config = json.loads((BASE / "modelos/clases.json").read_text(encoding="utf-8"))
    if modelo is None:
        modelo = tf.keras.models.load_model(BASE / "modelos/modelo_final.keras")
    imagen = cargar_imagen(ruta, config["lado"])
    valores = modelo(imagen[None, ...], training=False).numpy()[0]
    indice = int(np.argmax(valores))
    return {"escena": config["nombres"][indice], "clase": config["clases"][indice],
            "puntuacion": round(float(valores[indice]), 4),
            "puntuaciones": {nombre: round(float(valor), 4) for nombre, valor in zip(config["clases"], valores)}}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Se clasifica una fotografía en seis tipos de escena.")
    parser.add_argument("imagen", help="Ruta a una imagen JPG o PNG")
    args = parser.parse_args()
    print(json.dumps(predecir_imagen(args.imagen), ensure_ascii=False, indent=2))
