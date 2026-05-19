"""
Project mini-Argos v1 — Fase 4: Registro de Identidades
========================================================
Script independiente que permite registrar nuevos usuarios en la base
de datos de embeddings faciales.

Flujo:
    1. Pide el nombre del usuario por consola.
    2. Abre un diálogo de archivos (tkinter) para seleccionar en lote
       las fotografías del usuario (3-6+ fotos, con/sin lentes, etc.).
    3. Procesa cada imagen:
       a. Lee con OpenCV y convierte BGR -> RGB.
       b. Detecta y recorta el rostro con MTCNN.
       c. Extrae el embedding de 512 dimensiones con InceptionResnetV1.
    4. Promedia todos los embeddings válidos para generar un único
       "Embedding Maestro" que represente al usuario.
    5. Guarda (o actualiza) el registro en `rostros/embeddings.pt`.

Dependencias:
    pip install facenet-pytorch opencv-python torch
"""

import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DIRECTORIO_ROSTROS = Path(__file__).parent / "rostros"
ARCHIVO_EMBEDDINGS = DIRECTORIO_ROSTROS / "embeddings.pt"

# Extensiones de imagen aceptadas por OpenCV
_EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ---------------------------------------------------------------------------
# Inicialización de modelos
# ---------------------------------------------------------------------------

def inicializar_modelos() -> tuple:
    """
    Carga MTCNN (detección facial) e InceptionResnetV1 (embeddings)
    y los envía al dispositivo disponible (CUDA si hay GPU, sino CPU).

    MTCNN:
        - image_size=160: tamaño estándar de FaceNet para el crop.
        - margin=20: margen extra alrededor del rostro recortado para
          capturar contexto facial (frente, barbilla).
        - select_largest=True: si hay varios rostros, toma el más grande
          (asumimos que la foto es del usuario registrado).
        - post_process=True: normaliza la imagen recortada a [-1, 1].

    InceptionResnetV1:
        - pretrained='vggface2': pesos pre-entrenados en VGGFace2,
          excelente para reconocimiento facial general.
        - .eval(): modo evaluación (desactiva dropout y batch norm
          en modo entrenamiento).

    Returns:
        tuple: (mtcnn, modelo_embedding, dispositivo)
    """
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SISTEMA] Dispositivo seleccionado: {dispositivo}")

    print("[MODELO] Cargando MTCNN (detección facial)...")
    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        select_largest=True,
        post_process=True,
        device=dispositivo,
    )

    print("[MODELO] Cargando InceptionResnetV1 (embeddings)...")
    modelo_embedding = InceptionResnetV1(
        pretrained="vggface2",
    ).eval().to(dispositivo)

    print("[MODELO] Modelos cargados correctamente.\n")
    return mtcnn, modelo_embedding, dispositivo


# ---------------------------------------------------------------------------
# Entrada del usuario
# ---------------------------------------------------------------------------

def solicitar_nombre() -> str:
    """
    Pide el nombre del nuevo usuario por consola.

    Valida que no esté vacío y elimina espacios en los extremos.

    Returns:
        str: Nombre del usuario ingresado.
    """
    while True:
        nombre = input("Ingrese el nombre del nuevo usuario: ").strip()
        if nombre:
            return nombre
        print("[ERROR] El nombre no puede estar vacío. Intente de nuevo.\n")


# ---------------------------------------------------------------------------
# Selección de archivos
# ---------------------------------------------------------------------------

def seleccionar_imagenes() -> list[str]:
    """
    Abre un diálogo de selección múltiple de archivos (explorador de Windows).

    Filtra por extensiones de imagen comunes. Permite seleccionar en lote
    todas las fotos del participante.

    Returns:
        list[str]: Lista de rutas absolutas seleccionadas, o lista vacía
                   si el usuario canceló el diálogo.
    """
    # Crear y ocultar la ventana root de tkinter (solo necesitamos el diálogo)
    root = tk.Tk()
    root.withdraw()

    rutas = filedialog.askopenfilenames(
        title="Seleccione las fotografías del usuario",
        filetypes=[
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
            ("Todos los archivos", "*.*"),
        ],
    )

    root.destroy()
    return list(rutas)


# ---------------------------------------------------------------------------
# Procesamiento de imágenes
# ---------------------------------------------------------------------------

def procesar_imagenes(
    rutas: list[str],
    mtcnn: MTCNN,
    modelo_embedding: InceptionResnetV1,
    dispositivo: torch.device,
) -> list[torch.Tensor]:
    """
    Itera sobre las rutas de imagen, detecta el rostro y extrae el embedding.

    Pipeline por imagen:
        1. cv2.imread()         -> frame BGR
        2. BGR -> RGB           -> requerido por MTCNN / PIL
        3. mtcnn(img_rgb)       -> rostro recortado (160x160) normalizado, o None
        4. modelo(rostro)       -> embedding de 512 dimensiones

    Si MTCNN no detecta rostro en una imagen, la omite con un aviso.

    Args:
        rutas:            Lista de rutas absolutas a las imágenes.
        mtcnn:            Instancia de MTCNN ya cargada.
        modelo_embedding: Instancia de InceptionResnetV1 ya cargada.
        dispositivo:      torch.device (cuda o cpu).

    Returns:
        list[torch.Tensor]: Lista de embeddings válidos (cada uno de forma [512]).
    """
    embeddings = []

    for i, ruta in enumerate(rutas, start=1):
        nombre_archivo = Path(ruta).name
        print(f"  [{i}/{len(rutas)}] Procesando: {nombre_archivo}...", end=" ")

        # -- Validar extensión --------------------------------------------------
        extension = Path(ruta).suffix.lower()
        if extension not in _EXTENSIONES_VALIDAS:
            print(f"Extensión no soportada ({extension}). Saltando.")
            continue

        # -- Leer imagen con OpenCV ---------------------------------------------
        img_bgr = cv2.imread(ruta)
        if img_bgr is None:
            print("No se pudo leer la imagen. Saltando.")
            continue

        # -- Convertir BGR -> RGB -----------------------------------------------
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # -- Detectar y recortar rostro con MTCNN -------------------------------
        # mtcnn() devuelve un tensor (3, 160, 160) normalizado si detecta
        # rostro, o None si no encuentra ninguno.
        rostro = mtcnn(img_rgb)

        if rostro is None:
            print("No se detectó rostro.")
            print(
                f"[ADVERTENCIA] No se detectó rostro en la imagen: "
                f"{nombre_archivo}. Saltando..."
            )
            continue

        # -- Extraer embedding --------------------------------------------------
        # El modelo espera un batch: (N, 3, 160, 160)
        rostro_batch = rostro.unsqueeze(0).to(dispositivo)

        with torch.no_grad():
            embedding = modelo_embedding(rostro_batch)

        # embedding tiene forma (1, 512); lo aplanamos a (512,)
        embeddings.append(embedding.squeeze(0).cpu())
        print("Embedding extraído.")

    return embeddings


# ---------------------------------------------------------------------------
# Promedio y persistencia
# ---------------------------------------------------------------------------

def generar_embedding_maestro(embeddings: list[torch.Tensor]) -> torch.Tensor:
    """
    Promedia una lista de embeddings para crear un único vector representativo.

    El promedio fusiona las variaciones (con lentes, sin lentes, distintos
    ángulos) en un solo punto del espacio latente, lo que mejora la robustez
    del reconocimiento posterior.

    Args:
        embeddings: Lista de tensores de forma (512,).

    Returns:
        torch.Tensor: Embedding promedio de forma (512,).
    """
    # Apilar en un tensor (N, 512) y promediar sobre el eje 0
    stack = torch.stack(embeddings, dim=0)
    return torch.mean(stack, dim=0)


def guardar_registro(nombre: str, embedding: torch.Tensor) -> None:
    """
    Guarda o actualiza el embedding del usuario en `rostros/embeddings.pt`.

    El archivo almacena un diccionario {nombre: embedding_tensor}.
    Si el archivo ya existe, se carga y se añade/sobreescribe la entrada.
    Si no existe, se crea uno nuevo.

    Args:
        nombre:    Nombre del usuario (clave del diccionario).
        embedding: Tensor de forma (512,) con el embedding maestro.
    """
    # Crear directorio si no existe
    DIRECTORIO_ROSTROS.mkdir(parents=True, exist_ok=True)

    # Cargar diccionario existente o empezar vacío
    if ARCHIVO_EMBEDDINGS.exists():
        registros = torch.load(ARCHIVO_EMBEDDINGS, weights_only=False)
        print(f"[DB] Base de datos existente cargada ({len(registros)} registro(s)).")
    else:
        registros = {}
        print("[DB] No se encontró base de datos existente. Creando una nueva.")

    # Añadir o actualizar
    accion = "actualizado" if nombre in registros else "registrado"
    registros[nombre] = embedding

    # Persistir
    torch.save(registros, ARCHIVO_EMBEDDINGS)
    print(f"[DB] Usuario '{nombre}' {accion} exitosamente.")
    print(f"[DB] Base de datos guardada en: {ARCHIVO_EMBEDDINGS}")
    print(f"[DB] Total de usuarios registrados: {len(registros)}")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Flujo principal del script de registro de identidades.

    Secuencia:
        1. Inicializa MTCNN + InceptionResnetV1 (GPU si disponible).
        2. Pide el nombre del usuario.
        3. Abre diálogo para seleccionar fotografías.
        4. Procesa cada foto: detecta rostro -> extrae embedding.
        5. Promedia los embeddings -> guarda en rostros/embeddings.pt.
    """
    print("=" * 60)
    print("  mini Argos — Registro de Identidades (Fase 4)")
    print("=" * 60)
    print()

    # -- Paso 1: Cargar modelos -------------------------------------------------
    mtcnn, modelo_embedding, dispositivo = inicializar_modelos()

    # -- Paso 2: Nombre del usuario ---------------------------------------------
    nombre = solicitar_nombre()
    print(f"\n  Registrando usuario: '{nombre}'\n")

    # -- Paso 3: Seleccionar imágenes -------------------------------------------
    print("[ARCHIVO] Abriendo explorador de archivos...")
    rutas = seleccionar_imagenes()

    if not rutas:
        print("[ERROR] No se seleccionó ninguna imagen. Operación cancelada.")
        sys.exit(1)

    print(f"[ARCHIVO] {len(rutas)} imagen(es) seleccionada(s).\n")

    # -- Paso 4: Procesar imágenes ----------------------------------------------
    print("[PROCESO] Extrayendo embeddings faciales...\n")
    embeddings = procesar_imagenes(rutas, mtcnn, modelo_embedding, dispositivo)

    if not embeddings:
        print(
            "\n[ERROR] No se pudo extraer ningún embedding válido.\n"
            "        Ninguna de las imágenes seleccionadas contenía un\n"
            "        rostro detectable. Verifique la calidad de las fotos."
        )
        sys.exit(1)

    # -- Paso 5: Promediar y guardar --------------------------------------------
    print(f"\n[PROCESO] Generando Embedding Maestro a partir de "
          f"{len(embeddings)} embedding(s) válido(s)...")
    embedding_maestro = generar_embedding_maestro(embeddings)

    print()
    guardar_registro(nombre, embedding_maestro)

    # -- Paso 6: Resumen final --------------------------------------------------
    print()
    print("=" * 60)
    print(f"  REGISTRO EXITOSO")
    print(f"  Usuario:        {nombre}")
    print(f"  Fotos válidas:  {len(embeddings)} de {len(rutas)}")
    print(f"  Dimensión:      {embedding_maestro.shape[0]} (vector)")
    print(f"  Almacenado en:  {ARCHIVO_EMBEDDINGS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
