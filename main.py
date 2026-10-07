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
    allow_origins=["*"],  # En producción se puede restringir a 'http://biz.ht'
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. CARGA INICIAL DE DATOS Y MODELOS (Se ejecuta 1 sola vez al arrancar)
print("Cargando matrices de características y datos de catálogo...")

try:
    feature_matrix_gris = np.load(os.path.join(BASE_DIR, 'feature_matrix_gris.npy'))
    feature_matrix_color = np.load(os.path.join(BASE_DIR, 'feature_matrix_color.npy'))
    df = pd.read_csv(os.path.join(BASE_DIR, 'df_limpio.csv'))
except Exception as e:
    print(f"Error cargando archivos de datos: {e}")

# Limpieza y preparación de filtros en memoria
for col in ['style', 'pattern', 'colors']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({'nan': 'Unspecified', '': 'Unspecified'}).fillna('Unspecified')

df['color_base'] = df['colors'].str.split().str[0]

# Carga de la arquitectura ResNet-50 (extractor de características)
print("Cargando modelo ResNet-50...")
model = models.resnet50(pretrained=True)
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

print(" Backend listo para recibir peticiones.")


# 2. ENDPOINTS DE LA API

@app.get("/")
def home():
    """Endpoint de comprobación para verificar que la API está activa."""
    return {"status": "ok", "message": "API de Buscador Visual Nua Colección activa."}


@app.get("/filtros")
def obtener_filtros():
    """Devuelve las opciones únicas para poblar los 3 desplegables del sitio web."""
    return {
        "estilos": ["Todos"] + sorted(df['style'].unique().tolist()),
        "patrones": ["Todos"] + sorted(df['pattern'].unique().tolist()),
        "colores": ["Todos"] + sorted(df['color_base'].unique().tolist())
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
        # Lectura de la imagen enviada
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail="El archivo enviado no es una imagen válida.")

    # Filtrado previo según los 3 desplegables
    condicion = pd.Series([True] * len(df))
    if estilo != "Todos":
        condicion = condicion & (df['style'] == estilo)
    if patron != "Todos":
        condicion = condicion & (df['pattern'] == patron)
    if color != "Todos":
        condicion = condicion & (df['color_base'] == color)

    indices_validos = df[condicion].index.values

    # Si ningún artículo cumple con los filtros seleccionados
    if len(indices_validos) == 0:
        return {"estructura": [], "diseno": []}

    # Transformación del tensor de la imagen de consulta
    img_gris_t = transform_patron(img)
    img_color_t = transform_color(img)

    # Extracción de vectores con ResNet-50
    with torch.no_grad():
        q_v_gris = model(torch.unsqueeze(img_gris_t, 0)).flatten().numpy().reshape(1, -1)
        q_v_color = model(torch.unsqueeze(img_color_t, 0)).flatten().numpy().reshape(1, -1)

    # Definir el top K (máximo 10 resultados o el número de coincidencias disponibles)
    top_k = min(10, len(indices_validos))

    # A) Similitud para Estructura / Trama (Grayscale)
    sim_gris = cosine_similarity(q_v_gris, feature_matrix_gris[indices_validos]).flatten()
    top_gris_local = sim_gris.argsort()[-top_k:][::-1]
    top_10_gris_indices = indices_validos[top_gris_local]

    res_estructura = [
        {
            "articulo": str(df.iloc[i]['articulo']),
            "style": str(df.iloc[i]['style']),
            "pattern": str(df.iloc[i]['pattern']),
            "colors": str(df.iloc[i]['colors']),
            "imagen_url": f"http://biz.ht/{df.iloc[i]['tumb']}"
        }
        for i in top_10_gris_indices
    ]

    # B) Similitud para Diseño / Color (RGB)
    sim_color = cosine_similarity(q_v_color, feature_matrix_color[indices_validos]).flatten()
    top_color_local = sim_color.argsort()[-top_k:][::-1]
    top_10_color_indices = indices_validos[top_color_local]

    res_diseno = [
        {
            "articulo": str(df.iloc[i]['articulo']),
            "style": str(df.iloc[i]['style']),
            "pattern": str(df.iloc[i]['pattern']),
            "colors": str(df.iloc[i]['colors']),
            "imagen_url": f"http://biz.ht/{df.iloc[i]['tumb']}"
        }
        for i in top_10_color_indices
    ]

    return {
        "estructura": res_estructura,
        "diseno": res_diseno
    }
