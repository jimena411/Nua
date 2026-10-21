import os
import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Nua Colección - API Buscador Visual")

# Configurar CORS para permitir que la web hospedada en biz.ht consuma esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. CARGA INICIAL DE DATOS Y MODELOS (Se ejecuta 1 sola vez al arrancar)
print("Cargando matrices de características y datos de catálogo...")

# Cargar archivos de datos de forma directa
feature_matrix_gris = np.load(os.path.join(BASE_DIR, 'feature_matrix_gris.npy'))
feature_matrix_color = np.load(os.path.join(BASE_DIR, 'feature_matrix_color.npy'))

# Carga de catálogo (si existe df_limpio.csv lo usa, si no usa textiles.csv)
csv_path = os.path.join(BASE_DIR, 'df_limpio.csv')
if not os.path.exists(csv_path):
    csv_path = os.path.join(BASE_DIR, 'textiles.csv')

df = pd.read_csv(csv_path)

# Limpieza y preparación de filtros en memoria
for col in ['style', 'pattern', 'colors']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({'nan': 'Unspecified', '': 'Unspecified'}).fillna('Unspecified')

if 'colors' in df.columns:
    df['color_base'] = df['colors'].str.split().str[0]
else:
    df['color_base'] = 'Unspecified'

# Carga de la arquitectura ResNet-50 (Sintaxis moderna)
print("Cargando modelo ResNet-50...")
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model = nn.Sequential(*list(model.children())[:-1])
model.eval()

# Transformaciones de imagen
transform_patron = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_color = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Backend listo para recibir peticiones.")


# 2. ENDPOINTS DE LA API

@app.get("/")
def home():
    """Endpoint de comprobación para verificar que la API está activa."""
    return {"status": "ok", "message": "API de Buscador Visual Nua Colección activa."}


@app.get("/filtros")
def obtener_filtros():
    """Devuelve las opciones únicas para poblar los 3 desplegables del sitio web."""
    estilos = sorted(df['style'].unique().tolist()) if 'style' in df.columns else []
    patrones = sorted(df['pattern'].unique().tolist()) if 'pattern' in df.columns else []
    colores = sorted(df['color_base'].unique().tolist()) if 'color_base' in df.columns else []

    return {
        "estilos": ["Todos"] + estilos,
        "patrones": ["Todos"] + patrones,
        "colores": ["Todos"] + colores
    }


@app.post("/buscar")
async def buscar_coincidencias(
    file: UploadFile = File(...),
    estilo: str = Form("Todos"),
    patron: str = Form("Todos"),
    color: str = Form("Todos")
):
    """
    Recibe la muestra del usuario + los 3 filtros y devuelve los 10 mejores resultados 
    de Trama/Estructura y los 10 mejores de Color/Diseño.
    """
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail="El archivo enviado no es una imagen válida.")

    condicion = pd.Series([True] * len(df))
    if estilo != "Todos" and 'style' in df.columns:
        condicion = condicion & (df['style'] == estilo)
    if patron != "Todos" and 'pattern' in df.columns:
        condicion = condicion & (df['pattern'] == patron)
    if color != "Todos" and 'color_base' in df.columns:
        condicion = condicion & (df['color_base'] == color)

    indices_validos = df[condicion].index.values

    if len(indices_validos) == 0:
        return {"estructura": [], "diseno": []}

    img_gris_t = transform_patron(img)
    img_color_t = transform_color(img)

    with torch.no_grad():
        q_v_gris = model(torch.unsqueeze(img_gris_t, 0)).squeeze().numpy().reshape(1, -1)
        q_v_color = model(torch.unsqueeze(img_color_t, 0)).squeeze().numpy().reshape(1, -1)

    top_k = min(10, len(indices_validos))

    sim_gris = cosine_similarity(q_v_gris, feature_matrix_gris[indices_validos]).flatten()
    top_gris_local = sim_gris.argsort()[-top_k:][::-1]
    top_10_gris_indices = indices_validos[top_gris_local]

    res_estructura = [
        {
            "articulo": str(df.iloc[i]['articulo']),
            "style": str(df.iloc[i].get('style', 'Unspecified')),
            "pattern": str(df.iloc[i].get('pattern', 'Unspecified')),
            "colors": str(df.iloc[i].get('colors', 'Unspecified')),
            "imagen_url": f"http://biz.ht/{df.iloc[i]['tumb']}"
        }
        for i in top_10_gris_indices
    ]

    sim_color = cosine_similarity(q_v_color, feature_matrix_color[indices_validos]).flatten()
    top_color_local = sim_color.argsort()[-top_k:][::-1]
    top_10_color_indices = indices_validos[top_color_local]

    res_diseno = [
        {
            "articulo": str(df.iloc[i]['articulo']),
            "style": str(df.iloc[i].get('style', 'Unspecified')),
            "pattern": str(df.iloc[i].get('pattern', 'Unspecified')),
            "colors": str(df.iloc[i].get('colors', 'Unspecified')),
            "imagen_url": f"http://biz.ht/{df.iloc[i]['tumb']}"
        }
        for i in top_10_color_indices
    ]

    return {
        "estructura": res_estructura,
        "diseno": res_diseno
    }
