"""
Project mini-Argos v1 — Fase 7.4: UI + IA + Registro + Delay Anti-Spam + TTS Asíncrono
=========================================================================================
Módulo de interfaz gráfica del proyecto. Gestiona el login, control de cámaras
(RTMP + Dron), visualización en vivo (YOLOv8 + FaceNet), alertas con delay de 5s
para intrusos, alertas de voz tácticas (pyttsx3, no bloqueantes), ventana de
registro en monitor secundario y recarga thread-safe de la base de embeddings
sin reiniciar la app.
"""

import sys
import queue
import threading
import json
import winsound
import time
from pathlib import Path
from datetime import datetime

import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO

# ── Importaciones Estrictas de PyQt6 ───────────────────────────────────────
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QFrame, QScrollArea, QCheckBox,
    QStackedWidget, QSizePolicy, QSpacerItem, QGraphicsDropShadowEffect,
    QTextEdit, QFileDialog, QMessageBox, QGridLayout, QInputDialog, QListWidget, QListWidgetItem)
from PyQt6.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve, QThread, pyqtSignal, pyqtSlot, pyqtProperty, QRect, QPoint
from PyQt6.QtGui import QPixmap, QFont, QColor, QIcon, QImage, QPainter, QBrush, QPen


# ---------------------------------------------------------------------------
# Persistencia y Variables Globales de IA
# ---------------------------------------------------------------------------
CONFIG_FILE = Path(__file__).parent / "config.json"
CONFIG_KEY  = "rtmp_url"

def cargar_config() -> str:
    """Lee config.json y retorna la URL RTMP almacenada."""
    if CONFIG_FILE.exists():
        try:
            with CONFIG_FILE.open("r", encoding="utf-8") as f:
                datos = json.load(f)
            return datos.get(CONFIG_KEY, "")
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[ADVERTENCIA] No se pudo leer {CONFIG_FILE.name}: {exc}")
    return ""

def guardar_config(url: str) -> None:
    """Guarda la URL RTMP en config.json."""
    datos = {CONFIG_KEY: url}
    try:
        with CONFIG_FILE.open("w", encoding="utf-8") as f:
            json.dump(datos, f, indent=4, ensure_ascii=False)
        print(f"[CONFIG] URL guardada en '{CONFIG_FILE.name}'.")
    except OSError as exc:
        print(f"[ERROR] No se pudo guardar la configuración: {exc}")

# Área activa (de main.py)
_AREA_ACTUAL = "Area 1"

_UMBRAL_RECONOCIMIENTO = 0.8
DIR_ROSTROS = Path(__file__).parent / "rostros"
DIR_AVATARES = DIR_ROSTROS / "avatares"
_ARCHIVO_EMBEDDINGS = DIR_ROSTROS / "embeddings.pt"
AREAS_CONFIG_FILE = DIR_ROSTROS / "areas_config.json"
USUARIOS_CONFIG_FILE = DIR_ROSTROS / "usuarios_config.json"

def cargar_json(ruta: Path, default_data: dict) -> dict:
    if not ruta.parent.exists():
        ruta.parent.mkdir(parents=True, exist_ok=True)
    if not ruta.exists():
        with ruta.open("w", encoding="utf-8") as f:
            json.dump(default_data, f, indent=4, ensure_ascii=False)
        return default_data
    try:
        with ruta.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Al leer JSON {ruta.name}: {e}")
        return default_data

def guardar_json(ruta: Path, data: dict):
    if not ruta.parent.exists():
        ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def inicializar_archivos_fase74():
    DIR_AVATARES.mkdir(parents=True, exist_ok=True)
    areas_default = {"Area 1": {"descripcion": "RTMP Principal", "activa": True},
                     "Area 2": {"descripcion": "Cámara Dron", "activa": False}}
    usuarios_default = {"admin": {"avatar": "", "permisos": ["Area 1", "Area 2"]}}
    cargar_json(AREAS_CONFIG_FILE, areas_default)
    cargar_json(USUARIOS_CONFIG_FILE, usuarios_default)

# Ejecutar inicialización de archivos JSON al cargar el módulo
inicializar_archivos_fase74()

# Lock global para lectura/escritura thread-safe de embeddings.pt
_LOCK_EMBEDDINGS = threading.Lock()

# Delay (segundos) de gracia antes de reportar un intruso como alarma
_DELAY_INTRUSO_SEG = 5.0


# ---------------------------------------------------------------------------
# Constantes de diseño
# ---------------------------------------------------------------------------

# Paleta táctica
CLR_BG          = "#0A1110"    # Fondo principal (ventana)
CLR_PANEL       = "#142321"    # Fondo de contenedores/paneles
CLR_ACCENT      = "#00DC50"    # Color de acento (botones, toggles activos)
CLR_TEXT         = "#FFFFFF"    # Texto principal
CLR_TEXT_SEC     = "#A0AEC0"    # Texto secundario / placeholder
CLR_BORDER       = "#1E3330"   # Bordes sutiles
CLR_ERROR        = "#FF4D4D"   # Rojo para errores / alertas
CLR_INPUT_BG     = "#0D1B18"   # Fondo de inputs
CLR_HOVER        = "#00FF5C"   # Hover en botones
CLR_PANEL_HOVER  = "#1A2E2B"   # Hover en nav items

# Ruta al logo
LOGO_PATH = Path(__file__).parent / "recursos" / "imagenes" / "logo.png"

# Credenciales de prueba
_USUARIO_VALIDO = "admin"
_CLAVE_VALIDA   = "admin"


# ---------------------------------------------------------------------------
# Hoja de estilos global (QSS)
# ---------------------------------------------------------------------------

STYLESHEET = f"""
/* ── Base ─────────────────────────────────────────────────────── */
QMainWindow, QWidget {{
    background-color: {CLR_BG};
    color: {CLR_TEXT};
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
}}

/* ── Labels ───────────────────────────────────────────────────── */
QLabel {{
    color: {CLR_TEXT};
    background: transparent;
}}

QLabel#label-secondary {{
    color: {CLR_TEXT_SEC};
    font-size: 12px;
}}

QLabel#label-error {{
    color: {CLR_ERROR};
    font-size: 12px;
    font-weight: bold;
}}

QLabel#label-title {{
    font-size: 18px;
    font-weight: bold;
    color: {CLR_TEXT};
}}

QLabel#label-section {{
    font-size: 11px;
    font-weight: bold;
    color: {CLR_TEXT_SEC};
    text-transform: uppercase;
    letter-spacing: 1px;
}}

QLabel#label-stat-value {{
    font-size: 22px;
    font-weight: bold;
    color: {CLR_ACCENT};
}}

/* ── Inputs ───────────────────────────────────────────────────── */
QLineEdit {{
    background-color: {CLR_INPUT_BG};
    color: {CLR_TEXT};
    border: 1px solid {CLR_BORDER};
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 14px;
    selection-background-color: {CLR_ACCENT};
}}

QLineEdit:focus {{
    border: 1px solid {CLR_ACCENT};
}}

/* ── Botones ──────────────────────────────────────────────────── */
QPushButton#btn-primary {{
    background-color: {CLR_ACCENT};
    color: {CLR_BG};
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-size: 14px;
    font-weight: bold;
}}

QPushButton#btn-primary:hover {{
    background-color: {CLR_HOVER};
}}

QPushButton#btn-primary:pressed {{
    background-color: #00B843;
}}

QPushButton#btn-nav {{
    background-color: transparent;
    color: {CLR_TEXT_SEC};
    border: none;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 13px;
    text-align: left;
}}

QPushButton#btn-nav:hover {{
    background-color: {CLR_PANEL_HOVER};
    color: {CLR_TEXT};
}}

QPushButton#btn-nav:checked {{
    background-color: rgba(0, 220, 80, 0.15);
    color: {CLR_ACCENT};
    font-weight: bold;
}}

/* ── Paneles / Frames ─────────────────────────────────────────── */
QFrame#panel {{
    background-color: {CLR_PANEL};
    border-radius: 8px;
    border: 1px solid {CLR_BORDER};
}}

QFrame#panel-monitor {{
    background-color: {CLR_BG};
    border-radius: 8px;
    border: 1px solid {CLR_BORDER};
}}

/* ── ScrollArea ───────────────────────────────────────────────── */
QScrollArea {{
    background: transparent;
    border: none;
}}

QScrollBar:vertical {{
    background: {CLR_BG};
    width: 6px;
    border-radius: 3px;
}}

QScrollBar::handle:vertical {{
    background: {CLR_BORDER};
    border-radius: 3px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background: {CLR_ACCENT};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
"""


# ---------------------------------------------------------------------------
# Componente: Switch Deslizante (Pill Toggle)
# ---------------------------------------------------------------------------

class SwitchToggle(QCheckBox):
    """
    Interruptor tipo Switch Deslizante (Pill Toggle) moderno e interactivo.
    Reemplaza los Checkboxes tradicionales con cápsulas deslizantes animadas.
    """
    def __init__(self, text="", parent=None, active_color="#00DC50", bg_color="#2D3748", knob_color="#FFFFFF"):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._active_color = QColor(active_color)
        self._bg_color = QColor(bg_color)
        self._knob_color = QColor(knob_color)
        
        self._position = 3.0
        self.setMinimumHeight(24)
        
        # Conectar el cambio de estado para disparar la animación
        self.stateChanged.connect(self._on_state_changed)
        
        # Inicializar posición según estado
        if self.isChecked():
            self._position = 23.0
            
    def _on_state_changed(self, state):
        target = 23.0 if self.isChecked() else 3.0
        self.anim = QPropertyAnimation(self, b"position")
        self.anim.setDuration(150)
        self.anim.setStartValue(self._position)
        self.anim.setEndValue(target)
        self.anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.anim.start()
        
    @pyqtProperty(float)
    def position(self):
        return self._position
        
    @position.setter
    def position(self, pos):
        self._position = pos
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Track dimensions: 40px wide, 18px high
        track_w = 40
        track_h = 18
        track_y = (self.height() - track_h) // 2
        
        # Draw background track (pill shape)
        track_rect = QRect(0, track_y, track_w, track_h)
        radius = int(track_h / 2)
        
        # Determinar color según estado
        color = self._active_color if self.isChecked() else self._bg_color
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(track_rect, radius, radius)
        
        # Draw knob (white circle sliding inside)
        knob_size = 12
        knob_y = track_y + (track_h - knob_size) // 2
        knob_rect = QRect(int(self._position), knob_y, knob_size, knob_size)
        painter.setBrush(QBrush(self._knob_color))
        painter.drawEllipse(knob_rect)
        
        # Draw text (to the right of the switch)
        if self.text():
            painter.setPen(QColor(CLR_TEXT))
            # Ajustar fuente para que luzca limpia e integrada sobre el fondo del panel
            painter.setFont(self.font())
            text_rect = QRect(track_w + 10, 0, self.width() - track_w - 10, self.height())
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text())


# ---------------------------------------------------------------------------
# Hilo de Voz TTS Asíncrono (Anti-Congelamiento)
# ---------------------------------------------------------------------------

class HiloVoz(threading.Thread):
    """
    Motor Text-to-Speech no bloqueante usando pyttsx3 + queue.Queue.

    CRÍTICO (Windows COM): pyttsx3.init() se llama dentro de run() para que
    CoInitialize/CoUninitialize operen en el mismo hilo nativo. Hacerlo en
    __init__ provoca errores de COM cuando el hilo de Qt crea el objeto.

    Protocolo de parada: encolar None actúa como sentinel; detener() lo hace
    y luego bloquea hasta que el hilo termina (join con timeout de seguridad).
    """

    def __init__(self, cola: queue.Queue):
        super().__init__(daemon=True, name="HiloVoz")
        self._cola = cola

    def run(self):
        # ── Inicialización COM-safe: SIEMPRE dentro de run() ───────────────
        try:
            import pyttsx3
            engine = pyttsx3.init()
        except Exception as exc:
            print(f"[VOZ] No se pudo inicializar pyttsx3: {exc}. Sin alertas de voz.")
            return

        # ── Selección de voz en español ────────────────────────────────────
        voces = engine.getProperty('voices')
        voz_es = None
        _keywords = ('es-es', 'es-mx', 'es-us', 'spanish', 'español',
                     'helena', 'sabina', 'pablo', 'laura', 'jorge')
        for voz in voces:
            id_lower  = voz.id.lower()
            nom_lower = (voz.name or "").lower()
            if any(k in id_lower or k in nom_lower for k in _keywords):
                voz_es = voz
                break

        if voz_es:
            engine.setProperty('voice', voz_es.id)
            print(f"[VOZ] Voz española activa: {voz_es.name}")
        else:
            print("[VOZ] Voz española no encontrada; usando voz por defecto.")

        engine.setProperty('rate', 160)   # palabras por minuto
        print("[VOZ] Motor TTS listo.")

        # ── Bucle principal: siempre vivo, procesa cola ────────────────────
        while True:
            texto = self._cola.get()
            if texto is None:             # sentinel → salir
                break
            try:
                engine.say(texto)
                engine.runAndWait()
            except Exception as exc:
                print(f"[VOZ] Error al sintetizar: {exc}")

        try:
            engine.stop()
        except Exception:
            pass
        print("[VOZ] Motor TTS detenido.")

    def detener(self):
        """Encola el sentinel de parada y espera cierre limpio."""
        self._cola.put(None)
        self.join(timeout=4.0)


# ---------------------------------------------------------------------------
# Clase: Lector de Stream RTMP (Drenado Asíncrono de Buffer)
# ---------------------------------------------------------------------------

class LectorStream:
    """
    Lector de stream RTMP en un hilo secundario para evitar latencia acumulada.
    Conserva siempre únicamente el último frame leído de la cámara.
    """
    def __init__(self, url: str) -> None:
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self._detenido = False
        self._lock = threading.Lock()
        self._hilo = None

    @property
    def esta_abierto(self) -> bool:
        return self.cap.isOpened()

    def iniciar(self) -> None:
        self._hilo = threading.Thread(target=self._bucle_lectura, daemon=True)
        self._hilo.start()
        print("[HILO] Lector de stream iniciado en segundo plano.")

    def _bucle_lectura(self) -> None:
        while not self._detenido:
            ret, frame = self.cap.read()
            if not ret:
                self._detenido = True
                break
            with self._lock:
                self.frame = frame

    def leer(self):
        with self._lock:
            return self.frame

    def detener(self) -> None:
        self._detenido = True
        if self._hilo is not None:
            self._hilo.join(timeout=3.0)
        self.cap.release()
        print("[HILO] Lector de stream detenido y recurso liberado.")


# ---------------------------------------------------------------------------
# Funciones Auxiliares de FaceNet
# ---------------------------------------------------------------------------

def _cargar_base_rostros(dispositivo: torch.device) -> dict:
    if not _ARCHIVO_EMBEDDINGS.exists():
        print(
            "[FACENET] No se encontró base de datos de rostros.\n"
            f"          Ruta esperada: {_ARCHIVO_EMBEDDINGS}\n"
            "          El sistema funcionará en modo solo-detección."
        )
        return {}
    registros = torch.load(_ARCHIVO_EMBEDDINGS, weights_only=False)
    registros = {nombre: emb.to(dispositivo) for nombre, emb in registros.items()}
    print(f"[FACENET] Base de datos cargada: {len(registros)} usuario(s) registrado(s).")
    return registros

def _identificar_rostro(
    roi_bgr,
    mtcnn: MTCNN,
    modelo_facenet: InceptionResnetV1,
    base_rostros: dict,
    dispositivo: torch.device,
) -> str | None:
    if not base_rostros:
        return None
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    rostro = mtcnn(roi_rgb)
    if rostro is None:
        return None
    rostro_batch = rostro.unsqueeze(0).to(dispositivo)
    with torch.no_grad():
        embedding_actual = modelo_facenet(rostro_batch).squeeze(0)
    
    mejor_nombre = None
    mejor_distancia = float("inf")
    for nombre, embedding_reg in base_rostros.items():
        distancia = torch.dist(embedding_actual, embedding_reg, p=2).item()
        if distancia < mejor_distancia:
            mejor_distancia = distancia
            mejor_nombre = nombre
    if mejor_distancia < _UMBRAL_RECONOCIMIENTO:
        return mejor_nombre
    return None


# ---------------------------------------------------------------------------
# Hilo de Procesamiento IA (QThread)
# ---------------------------------------------------------------------------

class HiloProcesamientoIA(QThread):
    """
    Hilo secundario que encapsula la inferencia de YOLOv8, FaceNet,
    evaluación de permisos de acceso, emisión de alertas con delay anti-spam,
    alertas de voz tácticas (A/B/C) y alarmas sonoras en bucle.

    Delay de 5 segundos para INTRUSOS DESCONOCIDOS:
        - Antes de los 5s: recuadro AMARILLO, sin sonido ni log.
        - Tras 5s continuos: recuadro ROJO, alarma winsound + aviso de voz.
        - Si la zona se despeja: reinicia contador, bandera y alarma.

    Usuarios conocidos sin permiso de área: recuadro ROJO inmediato + voz
    con histéresis propia (sin alarma winsound).

    Recarga en caliente de embeddings:
        - recargar_embeddings() puede invocarse desde el hilo principal
          tras un registro nuevo; el hilo recarga embeddings.pt con el
          lock global para evitar lecturas/escrituras concurrentes.
    """
    signal_frame  = pyqtSignal(QImage)
    signal_alerta = pyqtSignal(str, str, bool)   # (nombre, area, autorizado)
    signal_stats  = pyqtSignal(int, int, float)  # (personas, nuevos, fps)
    signal_error  = pyqtSignal(str)

    def __init__(self, rtmp_url: str, cola_voz: queue.Queue, parent=None):
        super().__init__(parent)
        self.rtmp_url   = rtmp_url
        self._cola_voz  = cola_voz
        self._running   = True
        self.lector     = None
        # Flag thread-safe para solicitar recarga de embeddings
        self._reload_embeddings = False
        self._dispositivo = None   # se asigna al arrancar run()

    def run(self):
        self._dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SISTEMA] Dispositivo de inferencia: {self._dispositivo}")

        print("[YOLO] Cargando modelo YOLOv8n...")
        try:
            modelo_yolo = YOLO("yolov8n.pt")
        except Exception as e:
            self.signal_error.emit(f"Error al cargar YOLO: {e}")
            return

        print("[FACENET] Cargando MTCNN...")
        mtcnn = MTCNN(
            image_size=160,
            margin=20,
            select_largest=True,
            post_process=True,
            device=self._dispositivo,
        )

        print("[FACENET] Cargando InceptionResnetV1...")
        modelo_facenet = InceptionResnetV1(
            pretrained="vggface2",
        ).eval().to(self._dispositivo)

        with _LOCK_EMBEDDINGS:
            base_rostros = _cargar_base_rostros(self._dispositivo)
        
        usuarios_config = cargar_json(USUARIOS_CONFIG_FILE, {})
        lista_autorizados = []
        for uname, udata in usuarios_config.items():
            if _AREA_ACTUAL in udata.get("permisos", []):
                lista_autorizados.append(uname)

        self.lector = LectorStream(self.rtmp_url)
        if not self.lector.esta_abierto:
            self.signal_error.emit("No se pudo abrir el stream RTMP.")
            return
        self.lector.iniciar()

        alarma_activa    = False
        prev_time        = time.time()
        counter_fps      = 0
        fps              = 0.0
        alertas_recientes: dict = {}   # {nombre: timestamp}
        self.memoria_autorizados = {}

        # ── Estado del delay anti-spam ──────────────────────────────
        tiempo_inicio_desconocido: float | None = None

        # ── Estado TTS: histéresis y bandera de aviso de intruso ───
        memoria_no_autorizados: dict = {}   # {nombre: timestamp último aviso voz}
        advertencia_voz_dicha: bool  = False

        while self._running:
            if self._reload_embeddings:
                self._reload_embeddings = False
                with _LOCK_EMBEDDINGS:
                    base_rostros = _cargar_base_rostros(self._dispositivo)
                print("[IA] Base de rostros recargada en caliente.")

            if getattr(self, '_reload_config', False):
                with threading.Lock():
                    self._reload_config = False
                    usuarios_config = cargar_json(USUARIOS_CONFIG_FILE, {})
                    lista_autorizados.clear()
                    for uname, udata in usuarios_config.items():
                        if _AREA_ACTUAL in udata.get("permisos", []):
                            lista_autorizados.append(uname)
                    with _LOCK_EMBEDDINGS:
                        base_rostros = _cargar_base_rostros(self._dispositivo)
                    
                    # Vaciando memorias de tiempo
                    self.memoria_autorizados.clear()
                    alertas_recientes.clear()
                    memoria_no_autorizados.clear()
                    tiempo_inicio_desconocido = None
                    advertencia_voz_dicha = False
                    
                print("[IA] Configuración de permisos e IA recargada (Reinicio Táctico).")

            frame = self.lector.leer()
            if frame is None:
                if self.lector._detenido:
                    break
                self.msleep(30)
                continue

            curr_time = time.time()
            counter_fps += 1
            if curr_time - prev_time >= 1.0:
                fps = counter_fps / (curr_time - prev_time)
                counter_fps = 0
                prev_time = curr_time

            resultados = modelo_yolo(frame, classes=[0], verbose=False)
            cajas = resultados[0].boxes

            hay_no_autorizado_en_frame = False
            hay_desconocido_en_frame   = False
            personas_detectadas = len(cajas)

            if personas_detectadas > 0:
                for caja in cajas:
                    x1, y1, x2, y2 = caja.xyxy[0].int().tolist()
                    nombre_detectado = None
                    try:
                        alto_frame, ancho_frame = frame.shape[:2]
                        rx1 = max(0, x1)
                        ry1 = max(0, y1)
                        rx2 = min(ancho_frame, x2)
                        ry2 = min(alto_frame, y2)
                        roi = frame[ry1:ry2, rx1:rx2]
                        if roi.shape[0] > 30 and roi.shape[1] > 30:
                            nombre_detectado = _identificar_rostro(
                                roi, mtcnn, modelo_facenet,
                                base_rostros, self._dispositivo
                            )
                    except Exception as exc:
                        print(f"[ADVERTENCIA] Error en reconocimiento facial: {exc}")

                    # ── A / B / C: autorización + alertas de voz ──────────
                    ahora = time.time()

                    if nombre_detectado and nombre_detectado in lista_autorizados:
                        # ── A. AUTORIZADO ────────────────────────────────
                        color_caja = (80, 220, 0)           # Verde BGR
                        etiqueta   = f"{nombre_detectado} - AUTORIZADO"

                        ultimo_visto = self.memoria_autorizados.get(nombre_detectado, 0.0)
                        if (ahora - ultimo_visto) > 4.0:
                            print(f"[IA] {nombre_detectado} autorizado detectado.")
                            self.signal_alerta.emit(nombre_detectado, _AREA_ACTUAL, True)
                            self._cola_voz.put(f"{nombre_detectado}, reconocido.")
                        self.memoria_autorizados[nombre_detectado] = ahora

                    elif nombre_detectado:
                        # ── B. CONOCIDO PERO SIN PERMISO DE ÁREA ────────
                        hay_no_autorizado_en_frame = True
                        color_caja = (77, 77, 255)          # Rojo BGR
                        etiqueta   = f"{nombre_detectado} - NO AUTORIZADO"

                        if (ahora - alertas_recientes.get(nombre_detectado, 0.0)) > 15.0:
                            self.signal_alerta.emit(nombre_detectado, _AREA_ACTUAL, False)
                            alertas_recientes[nombre_detectado] = ahora

                        if (ahora - memoria_no_autorizados.get(nombre_detectado, 0.0)) > 15.0:
                            self._cola_voz.put(
                                f"{nombre_detectado} reconocido. "
                                "No está autorizado para estar en esta área."
                            )
                            memoria_no_autorizados[nombre_detectado] = ahora

                    else:
                        # ── C. COMPLETAMENTE DESCONOCIDO ────────────────
                        hay_no_autorizado_en_frame = True
                        hay_desconocido_en_frame   = True

                        if tiempo_inicio_desconocido is None:
                            tiempo_inicio_desconocido = ahora

                        segundos_intruso = ahora - tiempo_inicio_desconocido

                        if segundos_intruso >= _DELAY_INTRUSO_SEG:
                            color_caja = (77, 77, 255)      # Rojo BGR
                            etiqueta   = "Desconocido - NO AUTORIZADO"

                            if (ahora - alertas_recientes.get("Desconocido", 0.0)) > 15.0:
                                self.signal_alerta.emit("Desconocido", _AREA_ACTUAL, False)
                                alertas_recientes["Desconocido"] = ahora

                            # Aviso de voz: una sola vez por evento de intrusión
                            if not advertencia_voz_dicha:
                                self._cola_voz.put(
                                    "Usuario Desconocido, "
                                    "abandone el área o se dará alerta."
                                )
                                advertencia_voz_dicha = True
                        else:
                            color_caja = (0, 211, 255)      # Amarillo BGR
                            etiqueta   = "Desconocido [Evaluando...]"

                    # Pintar SIEMPRE
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_caja, 2)
                    (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color_caja, -1)
                    cv2.putText(
                        frame, etiqueta, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA
                    )

            if personas_detectadas > 0:
                cv2.putText(
                    frame, f"ESTADO: {personas_detectadas} Persona(s) detectada(s)",
                    (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 80), 2, cv2.LINE_AA
                )
            else:
                cv2.putText(
                    frame, "ESTADO: No hay personas",
                    (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 50, 235), 2, cv2.LINE_AA
                )

            self.signal_stats.emit(personas_detectadas, 0, fps)

            # ── Gestión de alarma (solo desconocidos reales) ───────────────
            if hay_desconocido_en_frame:
                if (tiempo_inicio_desconocido is not None and
                        time.time() - tiempo_inicio_desconocido >= _DELAY_INTRUSO_SEG):
                    if not alarma_activa:
                        alarma_activa = True
                        print("[ALARMA] Intruso desconocido confirmado tras 5s. Activando alarma...")
                        try:
                            winsound.PlaySound(
                                "SystemHand",
                                winsound.SND_ALIAS | winsound.SND_ASYNC | winsound.SND_LOOP,
                            )
                        except Exception:
                            pass
            else:
                # Zona despejada de desconocidos → reset completo
                tiempo_inicio_desconocido = None
                advertencia_voz_dicha = False
                if alarma_activa:
                    alarma_activa = False
                    print("[ALARMA] Zona despejada. Desactivando alarma.")
                    try:
                        winsound.PlaySound(None, winsound.SND_PURGE)
                    except Exception:
                        pass

            # Convertir frame a QImage y emitir a la UI
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qimg = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.signal_frame.emit(qimg.copy())

        # Cleanup
        if alarma_activa:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
        if self.lector:
            self.lector.detener()

    @pyqtSlot()
    def recargar_embeddings(self):
        self._reload_embeddings = True
        print("[IA] Recarga de embeddings solicitada.")

    @pyqtSlot()
    def recargar_configuracion(self):
        self._reload_config = True

    def stop(self):
        self._running = False
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass
        self.wait()


# ---------------------------------------------------------------------------
# Widget: Tarjeta de Alerta
# ---------------------------------------------------------------------------

class TarjetaAlerta(QFrame):
    """
    Widget que representa una alerta visual en el panel derecho.

    Muestra una línea de tiempo, nombre y estado con un borde lateral
    coloreado (verde = autorizado, rojo = no autorizado).
    """

    def __init__(
        self,
        hora: str,
        nombre: str,
        area: str,
        autorizado: bool,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("panel")

        color_borde = CLR_ACCENT if autorizado else CLR_ERROR
        estado = "Autorizado" if autorizado else "No Autorizado"

        self.setStyleSheet(f"""
            QFrame#panel {{
                background-color: {CLR_PANEL};
                border-left: 3px solid {color_borde};
                border-radius: 6px;
                border-top: 1px solid {CLR_BORDER};
                border-right: 1px solid {CLR_BORDER};
                border-bottom: 1px solid {CLR_BORDER};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        # Hora
        lbl_hora = QLabel(hora)
        lbl_hora.setObjectName("label-secondary")
        lbl_hora.setStyleSheet(f"font-size: 11px; color: {CLR_TEXT_SEC};")

        # Nombre + Estado
        lbl_nombre = QLabel(f"{nombre}")
        lbl_nombre.setStyleSheet("font-weight: bold; font-size: 13px;")

        lbl_estado = QLabel(f"{area}: {estado}")
        lbl_estado.setStyleSheet(f"color: {color_borde}; font-size: 12px;")

        layout.addWidget(lbl_hora)
        layout.addWidget(lbl_nombre)
        layout.addWidget(lbl_estado)


# ---------------------------------------------------------------------------
# Widget: Pantalla de Login
# ---------------------------------------------------------------------------

class PantallaLogin(QWidget):
    """
    Formulario de login centrado en pantalla con credenciales de prueba.

    Emite una señal visual (callback) al autenticarse correctamente.
    """

    def __init__(self, on_login_exitoso: callable, parent=None):
        super().__init__(parent)
        self._on_login_exitoso = on_login_exitoso
        self._setup_ui()

    def _setup_ui(self):
        # Layout principal centrado
        layout_outer = QVBoxLayout(self)
        layout_outer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Contenedor del formulario
        contenedor = QFrame()
        contenedor.setObjectName("panel")
        contenedor.setFixedSize(400, 480)

        # Sombra sutil
        sombra = QGraphicsDropShadowEffect()
        sombra.setBlurRadius(40)
        sombra.setOffset(0, 4)
        sombra.setColor(QColor(0, 0, 0, 120))
        contenedor.setGraphicsEffect(sombra)

        layout = QVBoxLayout(contenedor)
        layout.setContentsMargins(36, 36, 36, 36)
        layout.setSpacing(16)

        # Logo
        lbl_logo = QLabel()
        lbl_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if LOGO_PATH.exists():
            pixmap = QPixmap(str(LOGO_PATH))
            lbl_logo.setPixmap(
                pixmap.scaled(
                    90, 90,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        # Título
        lbl_titulo = QLabel("mini Argos")
        lbl_titulo.setObjectName("label-title")
        lbl_titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_titulo.setStyleSheet(
            f"font-size: 22px; font-weight: bold; color: {CLR_ACCENT};"
        )

        # Subtítulo
        lbl_sub = QLabel("Sistema de Vigilancia Inteligente")
        lbl_sub.setObjectName("label-secondary")
        lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Espaciador
        spacer_top = QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        # Campo: Usuario
        lbl_usuario = QLabel("Usuario")
        lbl_usuario.setStyleSheet(f"color: {CLR_TEXT_SEC}; font-size: 12px;")
        self.input_usuario = QLineEdit()
        self.input_usuario.setPlaceholderText("Ingrese su usuario")
        self.input_usuario.setMinimumHeight(42)

        # Campo: Contraseña
        lbl_clave = QLabel("Contraseña")
        lbl_clave.setStyleSheet(f"color: {CLR_TEXT_SEC}; font-size: 12px;")
        self.input_clave = QLineEdit()
        self.input_clave.setPlaceholderText("Ingrese su contraseña")
        self.input_clave.setEchoMode(QLineEdit.EchoMode.Password)
        self.input_clave.setMinimumHeight(42)

        # Enter para enviar
        self.input_clave.returnPressed.connect(self._intentar_login)

        # Label de error (oculto por defecto)
        self.lbl_error = QLabel("")
        self.lbl_error.setObjectName("label-error")
        self.lbl_error.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_error.setVisible(False)

        # Botón de login
        btn_login = QPushButton("Iniciar Sesión")
        btn_login.setObjectName("btn-primary")
        btn_login.setMinimumHeight(44)
        btn_login.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_login.clicked.connect(self._intentar_login)

        # Ensamblar
        layout.addWidget(lbl_logo)
        layout.addWidget(lbl_titulo)
        layout.addWidget(lbl_sub)
        layout.addItem(spacer_top)
        layout.addWidget(lbl_usuario)
        layout.addWidget(self.input_usuario)
        layout.addWidget(lbl_clave)
        layout.addWidget(self.input_clave)
        layout.addWidget(self.lbl_error)
        layout.addStretch()
        layout.addWidget(btn_login)

        layout_outer.addWidget(contenedor)

    def _intentar_login(self):
        """Valida credenciales y ejecuta callback si son correctas."""
        usuario = self.input_usuario.text().strip()
        clave = self.input_clave.text().strip()

        if usuario == _USUARIO_VALIDO and clave == _CLAVE_VALIDA:
            print(f"[LOGIN] Acceso concedido para: {usuario}")
            self.lbl_error.setVisible(False)
            self._on_login_exitoso()
        else:
            self.lbl_error.setText("⚠ Credenciales incorrectas")
            self.lbl_error.setVisible(True)
            print(f"[LOGIN] Acceso denegado para: {usuario}")


# ---------------------------------------------------------------------------
# Widget: Panel Izquierdo (Control)
# ---------------------------------------------------------------------------

class PanelControl(QFrame):
    """
    Columna izquierda del dashboard: logo, navegación, toggles de cámara
    y estadísticas en tiempo real (actualizadas por señales de la IA).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 20, 16, 20)
        layout.setSpacing(6)

        # ── Logo ──────────────────────────────────────────────────
        lbl_logo = QLabel()
        lbl_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if LOGO_PATH.exists():
            pixmap = QPixmap(str(LOGO_PATH))
            lbl_logo.setPixmap(
                pixmap.scaled(
                    64, 64,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        lbl_nombre = QLabel("mini Argos")
        lbl_nombre.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_nombre.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {CLR_ACCENT};"
        )

        lbl_version = QLabel("v1.0 — Sistema de Vigilancia")
        lbl_version.setObjectName("label-secondary")
        lbl_version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_version.setStyleSheet(f"font-size: 10px; color: {CLR_TEXT_SEC};")

        layout.addWidget(lbl_logo)
        layout.addWidget(lbl_nombre)
        layout.addWidget(lbl_version)

        # Separador
        layout.addSpacing(16)
        layout.addWidget(self._separador())
        layout.addSpacing(8)

        # ── Indicador de Estado "Argos" ───────────────────────────
        # Se ubica en la parte superior, justo arriba de navegación. Es más grande.
        self.lbl_argos = QLabel()
        self.lbl_argos.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_argos.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 8px;")
        self.actualizar_indicador_argos(False)
        layout.addWidget(self.lbl_argos)

        # ── Navegación ────────────────────────────────────────────
        lbl_nav = QLabel("NAVEGACIÓN")
        lbl_nav.setObjectName("label-section")

        self.btn_monitor = self._crear_boton_nav("🖥  Monitor en Vivo")
        self.btn_registro = self._crear_boton_nav("👤  Registro de Usuarios")
        self.btn_restricciones = self._crear_boton_nav("🔒  Restricciones de Usuario")
        self.btn_areas = self._crear_boton_nav("🗺  Gestión de Áreas")

        # Conectar señales
        self.btn_monitor.clicked.connect(
            lambda: print("[NAV] Monitor en Vivo")
        )
        self.btn_registro.clicked.connect(
            lambda: print("[NAV] Registro de Usuarios")
        )
        self.btn_restricciones.clicked.connect(
            lambda: print("[NAV] Restricciones de Usuario")
        )
        self.btn_areas.clicked.connect(
            lambda: print("[NAV] Gestión de Áreas")
        )

        # Marcar el primero como activo
        self.btn_monitor.setChecked(True)

        layout.addWidget(lbl_nav)
        layout.addWidget(self.btn_monitor)
        layout.addWidget(self.btn_registro)
        layout.addWidget(self.btn_restricciones)
        layout.addWidget(self.btn_areas)

        # Separador
        layout.addSpacing(12)
        layout.addWidget(self._separador())
        layout.addSpacing(8)

        # ── Controles de Cámara ───────────────────────────────────
        lbl_cam = QLabel("CONTROLES DE CÁMARA")
        lbl_cam.setObjectName("label-section")

        # Switches modernos de tipo Pill Toggle sin fondos oscuros
        self.chk_cam1 = SwitchToggle("Cámara 1 — RTMP")
        self.chk_cam1.setChecked(False)
        self.chk_cam2 = SwitchToggle("Cámara 2 — Dron")
        self.chk_cam2.setChecked(False)

        layout.addWidget(lbl_cam)
        layout.addWidget(self.chk_cam1)
        layout.addWidget(self.chk_cam2)

        # Separador
        layout.addSpacing(12)
        layout.addWidget(self._separador())
        layout.addSpacing(8)

        # ── Estadísticas ──────────────────────────────────────────
        lbl_stats = QLabel("ESTADÍSTICAS")
        lbl_stats.setObjectName("label-section")
        layout.addWidget(lbl_stats)

        self.lbl_stats = {}
        stats = [
            ("Personas", "0"),
            ("Nuevos", "0"),
            ("FPS", "0.0"),
        ]
        for nombre, valor in stats:
            w_stat, lbl_val = self._crear_stat(nombre, valor)
            layout.addWidget(w_stat)
            self.lbl_stats[nombre] = lbl_val

        layout.addStretch()

        # ── Pie: estado del sistema ───────────────────────────────
        lbl_estado = QLabel(f"● Sistema Activo")
        lbl_estado.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_estado.setStyleSheet(
            f"color: {CLR_ACCENT}; font-size: 11px; font-weight: bold;"
        )
        layout.addWidget(lbl_estado)

    def actualizar_indicador_argos(self, activo: bool):
        """Actualiza dinámicamente el texto del indicador de estado Argos."""
        if activo:
            self.lbl_argos.setText("Argos <font color='#00DC50'>Activo</font>")
        else:
            self.lbl_argos.setText("Argos <font color='#FF4D4D'>Desactivado</font>")

    def actualizar_stats(self, personas: int, nuevos: int, fps: float):
        """Actualiza dinámicamente las estadísticas del panel izquierdo."""
        if "Personas" in self.lbl_stats:
            self.lbl_stats["Personas"].setText(str(personas))
        if "Nuevos" in self.lbl_stats:
            self.lbl_stats["Nuevos"].setText(str(nuevos))
        if "FPS" in self.lbl_stats:
            self.lbl_stats["FPS"].setText(f"{fps:.1f}")

    def _crear_boton_nav(self, texto: str) -> QPushButton:
        """Crea un botón de navegación con estilo consistente."""
        btn = QPushButton(texto)
        btn.setObjectName("btn-nav")
        btn.setCheckable(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setMinimumHeight(38)
        return btn

    def _crear_stat(self, nombre: str, valor: str) -> tuple:
        """Crea un widget de estadística con nombre y valor."""
        contenedor = QWidget()
        contenedor.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(contenedor)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        lbl_nombre = QLabel(nombre)
        lbl_nombre.setStyleSheet(f"color: {CLR_TEXT_SEC}; font-size: 12px;")

        lbl_valor = QLabel(valor)
        lbl_valor.setObjectName("label-stat-value")
        lbl_valor.setAlignment(Qt.AlignmentFlag.AlignRight)

        layout.addWidget(lbl_nombre)
        layout.addStretch()
        layout.addWidget(lbl_valor)
        return contenedor, lbl_valor

    def _separador(self) -> QFrame:
        """Crea una línea horizontal sutil."""
        linea = QFrame()
        linea.setFrameShape(QFrame.Shape.HLine)
        linea.setStyleSheet(f"background-color: {CLR_BORDER}; max-height: 1px;")
        return linea


# ---------------------------------------------------------------------------
# Widget: Panel Central (Monitor en Vivo)
# ---------------------------------------------------------------------------

class PanelMonitor(QWidget):
    """
    Columna central del dashboard: muestra el stream en vivo (Cámara 1),
    el estado del dron (Cámara 2) o ambos en pantalla dividida vertical.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_update_time = 0.0  # Para la optimización de FPS

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Título
        header = QWidget()
        header.setStyleSheet("background: transparent;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(4, 0, 4, 0)

        lbl_titulo = QLabel("Monitor en Vivo")
        lbl_titulo.setObjectName("label-title")

        self.lbl_cam_activa = QLabel("● Sistema Inactivo")
        self.lbl_cam_activa.setStyleSheet(
            f"color: {CLR_TEXT_SEC}; font-size: 12px; font-weight: bold;"
        )

        header_layout.addWidget(lbl_titulo)
        header_layout.addStretch()
        header_layout.addWidget(self.lbl_cam_activa)

        layout.addWidget(header)

        # Contenedor principal de video
        self.frame_video = QFrame()
        self.frame_video.setObjectName("panel-monitor")
        self.frame_video.setMinimumHeight(400)
        self.frame_video.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )

        # Grid vertical táctico para apilar cámaras
        self.layout_split = QVBoxLayout(self.frame_video)
        self.layout_split.setContentsMargins(8, 8, 8, 8)
        self.layout_split.setSpacing(8)

        # ── Cámara 1: Video en Vivo ────────────────────────────────
        self.lbl_video_cam1 = QLabel()
        self.lbl_video_cam1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_video_cam1.setStyleSheet(f"background-color: #000000; border-radius: 6px; border: 1px solid {CLR_BORDER};")
        # Política de tamaño Ignored para evitar que el pixmap altere el tamaño o ancho de los paneles laterales
        self.lbl_video_cam1.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.lbl_video_cam1.setScaledContents(False)
        self.lbl_video_cam1.setVisible(False)

        # ── Cámara 2: Placeholder del Dron ─────────────────────────
        self.frame_drone = QFrame()
        self.frame_drone.setStyleSheet(f"""
            QFrame {{
                background-color: #050A09; 
                border-radius: 6px; 
                border: 1px solid {CLR_BORDER};
            }}
        """)
        # Política de tamaño Ignored para distribución táctica y simétrica simétrica 50/50
        self.frame_drone.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.frame_drone.setVisible(False)

        drone_layout = QVBoxLayout(self.frame_drone)
        drone_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lbl_drone_icono = QLabel("🛸")
        lbl_drone_icono.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_drone_icono.setStyleSheet("font-size: 40px; background: transparent;")
        
        lbl_drone_titulo = QLabel("CÁMARA 2 — DRON")
        lbl_drone_titulo.setObjectName("label-section")
        lbl_drone_titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lbl_drone_desc = QLabel("El dron está inactivo")
        lbl_drone_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_drone_desc.setStyleSheet(f"color: {CLR_ERROR}; font-size: 14px; font-weight: bold; background: transparent;")

        drone_layout.addWidget(lbl_drone_icono)
        drone_layout.addWidget(lbl_drone_titulo)
        drone_layout.addWidget(lbl_drone_desc)

        # ── General Placeholder (Ninguna cámara activa) ─────────────
        self.widget_placeholder = QWidget()
        self.widget_placeholder.setStyleSheet("background: transparent;")
        placeholder_layout = QVBoxLayout(self.widget_placeholder)
        placeholder_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lbl_icono = QLabel("📡")
        lbl_icono.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_icono.setStyleSheet("font-size: 48px; background: transparent;")

        self.lbl_placeholder = QLabel("Esperando señal de video...")
        self.lbl_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_placeholder.setStyleSheet(
            f"color: {CLR_TEXT_SEC}; font-size: 16px; background: transparent;"
        )

        lbl_hint = QLabel("Active una cámara desde el panel de control")
        lbl_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_hint.setStyleSheet(
            f"color: {CLR_BORDER}; font-size: 12px; background: transparent;"
        )

        placeholder_layout.addWidget(lbl_icono)
        placeholder_layout.addWidget(self.lbl_placeholder)
        placeholder_layout.addWidget(lbl_hint)

        # Añadir widgets al split layout
        self.layout_split.addWidget(self.lbl_video_cam1)
        self.layout_split.addWidget(self.frame_drone)
        self.layout_split.addWidget(self.widget_placeholder)

        layout.addWidget(self.frame_video)

    def actualizar_vistas(self, cam1_on: bool, cam2_on: bool):
        """Ajusta dinámicamente el layout según las cámaras encendidas (disposición táctica)."""
        if not cam1_on and not cam2_on:
            self.widget_placeholder.show()
            self.lbl_video_cam1.hide()
            self.frame_drone.hide()
            self.lbl_cam_activa.setText("● Sistema Inactivo")
            self.lbl_cam_activa.setStyleSheet(f"color: {CLR_TEXT_SEC}; font-size: 12px; font-weight: bold;")
        else:
            self.widget_placeholder.hide()
            if cam1_on:
                self.lbl_video_cam1.show()
            else:
                self.lbl_video_cam1.hide()

            if cam2_on:
                self.frame_drone.show()
            else:
                self.frame_drone.hide()
            
            # Actualizar cabecera de estado
            if cam1_on and cam2_on:
                self.lbl_cam_activa.setText("● Cámara 1 + Dron Activos")
                self.lbl_cam_activa.setStyleSheet(f"color: {CLR_ACCENT}; font-size: 12px; font-weight: bold;")
            elif cam1_on:
                self.lbl_cam_activa.setText("● Cámara 1 Activa")
                self.lbl_cam_activa.setStyleSheet(f"color: {CLR_ACCENT}; font-size: 12px; font-weight: bold;")
            else:
                self.lbl_cam_activa.setText("● Dron Activo (Cámara 2)")
                self.lbl_cam_activa.setStyleSheet(f"color: {CLR_ACCENT}; font-size: 12px; font-weight: bold;")

    def mostrar_frame(self, qimage: QImage):
        """Pinta el QImage procesado por el hilo de IA en la pantalla de la Cámara 1."""
        ahora = time.time()
        # Optimización de FPS: Evita saturar la cola de eventos limitando el refresco a ~30 FPS
        if ahora - self._last_update_time < 0.033:
            return

        if self.lbl_video_cam1.isVisible() and not qimage.isNull():
            w = self.lbl_video_cam1.width()
            h = self.lbl_video_cam1.height()
            if w > 0 and h > 0:
                self._last_update_time = ahora
                pixmap = QPixmap.fromImage(qimage)
                # Redimensionamiento dinámico thread-safe respetando estrictamente la relación de aspecto original
                scaled_pixmap = pixmap.scaled(
                    self.lbl_video_cam1.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.lbl_video_cam1.setPixmap(scaled_pixmap)

    def limpiar_monitor(self):
        """Limpia el visor de video y restablece la imagen oscura."""
        self.lbl_video_cam1.clear()


# ---------------------------------------------------------------------------
# Widget: Panel Derecho (Alertas)
# ---------------------------------------------------------------------------

class PanelAlertas(QFrame):
    """
    Columna derecha del dashboard: lista scrollable de tarjetas de alerta.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 20, 16, 16)
        layout.setSpacing(12)

        # Título
        lbl_titulo = QLabel("Panel de Alertas")
        lbl_titulo.setObjectName("label-title")
        lbl_titulo.setStyleSheet("font-size: 16px;")
        layout.addWidget(lbl_titulo)

        # Subtítulo
        lbl_sub = QLabel("Eventos recientes del sistema")
        lbl_sub.setObjectName("label-secondary")
        lbl_sub.setStyleSheet(f"font-size: 11px; color: {CLR_TEXT_SEC};")
        layout.addWidget(lbl_sub)

        layout.addSpacing(4)

        # Área con scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        # Contenedor interno del scroll
        contenido = QWidget()
        contenido.setStyleSheet("background: transparent;")
        self._layout_alertas = QVBoxLayout(contenido)
        self._layout_alertas.setContentsMargins(0, 0, 0, 0)
        self._layout_alertas.setSpacing(8)
        self._layout_alertas.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll.setWidget(contenido)
        layout.addWidget(scroll)

        # (HOTFIX) Inicio limpio: No agregamos alertas quemadas.

    def _agregar_alertas_demo(self):
        """Inserta tarjetas de alerta de prueba."""
        alertas_demo = [
            ("14:05:00", "Rony Carrillo", "Área 1", True),
            ("14:05:12", "Desconocido #1", "Área 1", False),
            ("14:06:30", "Maria Lopez", "Área 1", True),
            ("14:07:45", "Rony Carrillo", "Área 2", False),
            ("14:08:10", "Juan Perez", "Área 1", True),
            ("14:10:22", "Desconocido #2", "Área 1", False),
        ]

        for hora, nombre, area, autorizado in alertas_demo:
            tarjeta = TarjetaAlerta(hora, nombre, area, autorizado)
            self._layout_alertas.addWidget(tarjeta)

    def agregar_alerta(self, nombre: str, area: str, autorizado: bool):
        """
        Método público para añadir alertas dinámicamente (uso futuro).

        Inserta la tarjeta al inicio del scroll para que las más
        recientes aparezcan primero.
        """
        hora = datetime.now().strftime("%H:%M:%S")
        tarjeta = TarjetaAlerta(hora, nombre, area, autorizado)
        self._layout_alertas.insertWidget(0, tarjeta)


# ---------------------------------------------------------------------------
# Hilo de Registro (CPU-only, protegido con lock global)
# ---------------------------------------------------------------------------

class HiloRegistro(QThread):
    """
    Ejecuta el pipeline de extracción de embeddings faciales ESTRICTAMENTE
    en CPU para no saturar la VRAM que usa el HiloProcesamientoIA en GPU.

    Señales:
        signal_log(str)        — línea de progreso para el QTextEdit.
        signal_ok(dict)        — diccionario completo actualizado al finalizar.
        signal_error(str)      — mensaje de error si algo falla.
    """
    signal_log   = pyqtSignal(str)
    signal_ok    = pyqtSignal(dict)   # {nombre: embedding, ...}
    signal_error = pyqtSignal(str)

    def __init__(self, nombre: str, rutas: list[str], parent=None):
        super().__init__(parent)
        self.nombre = nombre
        self.rutas  = rutas

    def run(self):
        import cv2 as _cv2
        # ── REGLA CRÍTICA: CPU únicamente para proteger la VRAM ────────
        device_cpu = torch.device("cpu")
        self.signal_log.emit(f"[REGISTRO] Usando dispositivo: {device_cpu}")

        self.signal_log.emit("[REGISTRO] Cargando MTCNN (CPU)...")
        try:
            mtcnn_cpu = MTCNN(
                image_size=160, margin=20,
                select_largest=True, post_process=True,
                device=device_cpu,
            )
            modelo_cpu = InceptionResnetV1(
                pretrained="vggface2",
            ).eval().to(device_cpu)
        except Exception as e:
            self.signal_error.emit(f"Error al cargar modelos: {e}")
            return

        self.signal_log.emit("[REGISTRO] Modelos FaceNet (CPU) cargados.")

        embeddings_validos: list[torch.Tensor] = []
        avatar_guardado = False
        ruta_avatar = None

        for i, ruta in enumerate(self.rutas, 1):
            nombre_archivo = Path(ruta).name
            self.signal_log.emit(f"  [{i}/{len(self.rutas)}] Procesando: {nombre_archivo}...")

            img_bgr = _cv2.imread(ruta)
            if img_bgr is None:
                self.signal_log.emit(f"  ⚠ No se pudo leer: {nombre_archivo}")
                continue

            img_rgb = _cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2RGB)

            try:
                rostro = mtcnn_cpu(img_rgb)
            except Exception as exc:
                self.signal_log.emit(f"  ⚠ Error MTCNN: {exc}")
                continue

            if rostro is None:
                self.signal_log.emit(
                    f"  [ADVERTENCIA] No se detectó rostro en: {nombre_archivo}. Saltando..."
                )
                continue

            if not avatar_guardado:
                import torchvision.transforms as T
                img_pil = T.ToPILImage()((rostro + 1) / 2)
                ruta_avatar = DIR_AVATARES / f"{self.nombre.replace(' ', '_')}.jpg"
                img_pil.save(ruta_avatar)
                avatar_guardado = True
                self.signal_log.emit(f"  ✓ Avatar guardado en {ruta_avatar.name}")

            with torch.no_grad():
                emb = modelo_cpu(rostro.unsqueeze(0)).squeeze(0)
            embeddings_validos.append(emb.cpu())
            self.signal_log.emit(f"  ✓ Embedding extraído de {nombre_archivo}")

        if not embeddings_validos:
            self.signal_error.emit(
                "No se extrajo ningún embedding válido.\n"
                "Ninguna imagen contenía un rostro detectable."
            )
            return

        self.signal_log.emit(
            f"\n[REGISTRO] Generando Embedding Maestro con "
            f"{len(embeddings_validos)} embedding(s)..."
        )
        stack   = torch.stack(embeddings_validos, dim=0)
        maestro = torch.mean(stack, dim=0)

        with _LOCK_EMBEDDINGS:
            if _ARCHIVO_EMBEDDINGS.exists():
                registros = torch.load(
                    _ARCHIVO_EMBEDDINGS, weights_only=False
                )
            else:
                registros = {}
                _ARCHIVO_EMBEDDINGS.parent.mkdir(parents=True, exist_ok=True)

            accion = "actualizado" if self.nombre in registros else "registrado"
            registros[self.nombre] = maestro
            torch.save(registros, _ARCHIVO_EMBEDDINGS)

        usuarios_config = cargar_json(USUARIOS_CONFIG_FILE, {})
        if self.nombre not in usuarios_config:
            usuarios_config[self.nombre] = {
                "avatar": str(ruta_avatar.name) if avatar_guardado else "",
                "permisos": []
            }
        else:
            if avatar_guardado:
                usuarios_config[self.nombre]["avatar"] = str(ruta_avatar.name)
        guardar_json(USUARIOS_CONFIG_FILE, usuarios_config)

        self.signal_log.emit(
            f"\n[DB] Usuario '{self.nombre}' {accion} exitosamente."
        )
        self.signal_log.emit(
            f"[DB] Total usuarios en base de datos: {len(registros)}"
        )
        self.signal_ok.emit(registros)


# ---------------------------------------------------------------------------
# Ventana de Registro de Usuarios (Multi-Monitor, CPU-only FaceNet)
# ---------------------------------------------------------------------------

class VentanaRegistro(QMainWindow):
    """
    Ventana independiente para el registro de nuevos usuarios.

    Comportamiento multi-monitor:
        · Si hay > 1 monitor: se abre en pantalla completa en el monitor
          secundario (índice 1), dejando el dashboard en el principal.
        · Si hay solo 1 monitor: pantalla completa en el único disponible.

    El procesamiento se realiza en HiloRegistro (CPU-only) para no
    saturar la VRAM de la GPU que usa HiloProcesamientoIA.
    """

    # Señal emitida al guardar un nuevo usuario correctamente;
    # el Dashboard la conecta a hilo_ia.recargar_embeddings()
    registro_exitoso = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("mini Argos — Registro de Usuarios")
        self.setStyleSheet(QApplication.instance().styleSheet())
        self._rutas_seleccionadas: list[str] = []
        self._hilo_registro: HiloRegistro | None = None

        self._setup_ui()
        self._posicionar_en_monitor_secundario()

    def _posicionar_en_monitor_secundario(self):
        """Si hay un segundo monitor, abre la ventana en él en pantalla completa."""
        pantallas = QApplication.screens()
        if len(pantallas) > 1:
            geometria = pantallas[1].geometry()
            self.move(geometria.topLeft())
            print(f"[REGISTRO] Abriendo en monitor secundario: {geometria}")
        else:
            print("[REGISTRO] Solo 1 monitor detectado. Abriendo en pantalla principal.")
        self.showFullScreen()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(48, 48, 48, 48)
        outer.setSpacing(0)

        # ── Barra superior: logo + título + botón cerrar ───────────
        barra = QWidget()
        barra.setStyleSheet("background: transparent;")
        barra_layout = QHBoxLayout(barra)
        barra_layout.setContentsMargins(0, 0, 0, 16)

        if LOGO_PATH.exists():
            lbl_logo = QLabel()
            px = QPixmap(str(LOGO_PATH)).scaled(
                40, 40,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            lbl_logo.setPixmap(px)
            barra_layout.addWidget(lbl_logo)

        lbl_titulo = QLabel("Registro de Usuarios")
        lbl_titulo.setStyleSheet(
            f"font-size: 22px; font-weight: bold; color: {CLR_TEXT};"
        )
        barra_layout.addWidget(lbl_titulo)
        barra_layout.addStretch()

        btn_cerrar = QPushButton("✕  Volver al Dashboard")
        btn_cerrar.setObjectName("btn-primary")
        btn_cerrar.setStyleSheet(
            f"background: transparent; color: {CLR_TEXT_SEC};"
            f"border: 1px solid {CLR_BORDER}; border-radius: 8px;"
            f"padding: 8px 16px;"
        )
        btn_cerrar.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_cerrar.clicked.connect(self.close)
        barra_layout.addWidget(btn_cerrar)
        outer.addWidget(barra)

        # ── Layout de 2 columnas: formulario | consola ─────────────
        cols = QHBoxLayout()
        cols.setSpacing(24)
        outer.addLayout(cols)

        # ─── Columna izquierda: formulario ─────────────────────────
        col_form = QFrame()
        col_form.setObjectName("panel")
        col_form.setMaximumWidth(520)
        form_layout = QVBoxLayout(col_form)
        form_layout.setContentsMargins(32, 32, 32, 32)
        form_layout.setSpacing(16)

        lbl_sec = QLabel("DATOS DEL USUARIO")
        lbl_sec.setObjectName("label-section")
        form_layout.addWidget(lbl_sec)

        # Campo nombre
        lbl_nombre = QLabel("Nombre completo")
        lbl_nombre.setStyleSheet(f"color: {CLR_TEXT_SEC}; font-size: 12px;")
        self.input_nombre = QLineEdit()
        self.input_nombre.setPlaceholderText("Ej: Rony Carrillo")
        self.input_nombre.setMinimumHeight(44)
        form_layout.addWidget(lbl_nombre)
        form_layout.addWidget(self.input_nombre)

        form_layout.addSpacing(8)

        # Botón seleccionar fotos
        lbl_fotos = QLabel("Fotografías del usuario (3-6 recomendadas)")
        lbl_fotos.setStyleSheet(f"color: {CLR_TEXT_SEC}; font-size: 12px;")
        form_layout.addWidget(lbl_fotos)

        self.btn_seleccionar = QPushButton("📂  Seleccionar Fotografías")
        self.btn_seleccionar.setObjectName("btn-primary")
        self.btn_seleccionar.setStyleSheet(
            f"background-color: {CLR_PANEL}; color: {CLR_TEXT};"
            f"border: 1px solid {CLR_ACCENT}; border-radius: 8px;"
            f"padding: 12px 20px; font-size: 13px;"
        )
        self.btn_seleccionar.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_seleccionar.clicked.connect(self._seleccionar_fotos)
        form_layout.addWidget(self.btn_seleccionar)

        self.lbl_fotos_count = QLabel("Ninguna fotografía seleccionada")
        self.lbl_fotos_count.setStyleSheet(
            f"color: {CLR_TEXT_SEC}; font-size: 12px;"
        )
        form_layout.addWidget(self.lbl_fotos_count)

        self.grid_fotos = QGridLayout()
        form_layout.addLayout(self.grid_fotos)

        form_layout.addStretch()

        # Botón procesar (acento verde grande)
        self.btn_procesar = QPushButton("🧠  Procesar y Guardar")
        self.btn_procesar.setObjectName("btn-primary")
        self.btn_procesar.setMinimumHeight(52)
        self.btn_procesar.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_procesar.clicked.connect(self._iniciar_procesamiento)
        form_layout.addWidget(self.btn_procesar)

        cols.addWidget(col_form)

        # ─── Columna derecha: consola de progreso ──────────────────
        col_consola = QFrame()
        col_consola.setObjectName("panel")
        consola_layout = QVBoxLayout(col_consola)
        consola_layout.setContentsMargins(20, 20, 20, 20)
        consola_layout.setSpacing(8)

        lbl_consola = QLabel("PROGRESO")
        lbl_consola.setObjectName("label-section")
        consola_layout.addWidget(lbl_consola)

        self.consola = QTextEdit()
        self.consola.setReadOnly(True)
        self.consola.setStyleSheet(
            f"background-color: {CLR_INPUT_BG};"
            f"color: {CLR_ACCENT};"
            f"border: 1px solid {CLR_BORDER};"
            f"border-radius: 8px;"
            f"font-family: 'Consolas', monospace;"
            f"font-size: 12px;"
            f"padding: 12px;"
        )
        consola_layout.addWidget(self.consola)
        cols.addWidget(col_consola, stretch=1)

    # ── Atajos de teclado ──────────────────────────────────────────
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    # ── Lógica de UI ──────────────────────────────────────────────
    def _seleccionar_fotos(self):
        rutas, _ = QFileDialog.getOpenFileNames(
            self,
            "Seleccione las fotografías del usuario",
            "",
            "Imágenes (*.jpg *.jpeg *.png *.bmp *.webp *.tiff)"
        )
        if rutas:
            self._rutas_seleccionadas = rutas
            self.lbl_fotos_count.setText(
                f"✓ {len(rutas)} fotografía(s) seleccionada(s)"
            )
            self.lbl_fotos_count.setStyleSheet(
                f"color: {CLR_ACCENT}; font-size: 12px; font-weight: bold;"
            )
            # Limpiar grid
            for i in reversed(range(self.grid_fotos.count())): 
                widget_to_remove = self.grid_fotos.itemAt(i).widget()
                if widget_to_remove is not None:
                    widget_to_remove.setParent(None)
            
            # Poblar grid con miniaturas (hasta 6)
            for i, ruta in enumerate(rutas[:6]):
                lbl_thumb = QLabel()
                pix = QPixmap(ruta).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
                lbl_thumb.setPixmap(pix)
                lbl_thumb.setFixedSize(100, 100)
                lbl_thumb.setStyleSheet(f"border: 1px solid {CLR_BORDER}; border-radius: 4px;")
                row = i // 3
                col = i % 3
                self.grid_fotos.addWidget(lbl_thumb, row, col)
        else:
            self._rutas_seleccionadas = []
            self.lbl_fotos_count.setText("Ninguna fotografía seleccionada")
            self.lbl_fotos_count.setStyleSheet(
                f"color: {CLR_TEXT_SEC}; font-size: 12px;"
            )
            for i in reversed(range(self.grid_fotos.count())): 
                widget_to_remove = self.grid_fotos.itemAt(i).widget()
                if widget_to_remove is not None:
                    widget_to_remove.setParent(None)

    def _iniciar_procesamiento(self):
        """Valida los campos y lanza HiloRegistro."""
        nombre = self.input_nombre.text().strip()
        if not nombre:
            QMessageBox.warning(self, "Campo vacío", "Ingrese el nombre del usuario.")
            return
        if not self._rutas_seleccionadas:
            QMessageBox.warning(self, "Sin fotografías",
                                "Seleccione al menos una fotografía.")
            return

        # Deshabilitar controles mientras procesa
        self.btn_procesar.setEnabled(False)
        self.btn_seleccionar.setEnabled(False)
        self.consola.clear()
        self.consola.append(
            f"Iniciando registro de '{nombre}'...\n"
        )

        self._hilo_registro = HiloRegistro(nombre, self._rutas_seleccionadas)
        self._hilo_registro.signal_log.connect(self._on_log)
        self._hilo_registro.signal_ok.connect(self._on_registro_ok)
        self._hilo_registro.signal_error.connect(self._on_registro_error)
        self._hilo_registro.start()

    @pyqtSlot(str)
    def _on_log(self, mensaje: str):
        self.consola.append(mensaje)

    @pyqtSlot(dict)
    def _on_registro_ok(self, registros: dict):
        self.consola.append(
            f"\n✅ ¡Registro completado con éxito!"
        )
        self.btn_procesar.setEnabled(True)
        self.btn_seleccionar.setEnabled(True)
        # Notificar al Dashboard para que recargue la base de embeddings en el hilo IA
        self.registro_exitoso.emit()

    @pyqtSlot(str)
    def _on_registro_error(self, error: str):
        self.consola.append(f"\n❌ ERROR: {error}")
        self.btn_procesar.setEnabled(True)
        self.btn_seleccionar.setEnabled(True)
        QMessageBox.critical(self, "Error de Registro", error)


# ---------------------------------------------------------------------------
# Pantalla de Permisos (Fase 7.4)
# ---------------------------------------------------------------------------

class PantallaPermisos(QWidget):
    cambios_guardados = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # ── Columna Izquierda (Directorio) ──
        self.panel_izq = QFrame()
        self.panel_izq.setObjectName("panel")
        izq_layout = QVBoxLayout(self.panel_izq)
        
        lbl_dir = QLabel("DIRECTORIO")
        lbl_dir.setObjectName("label-section")
        izq_layout.addWidget(lbl_dir)
        
        self.lista_usuarios = QListWidget()
        self.lista_usuarios.setStyleSheet(f"background: transparent; border: none; font-size: 14px; outline: none;")
        self.lista_usuarios.itemClicked.connect(self._on_usuario_seleccionado)
        izq_layout.addWidget(self.lista_usuarios)
        
        layout.addWidget(self.panel_izq, stretch=3)

        # ── Columna Derecha (Panel de Control) ──
        self.panel_der = QFrame()
        self.panel_der.setObjectName("panel")
        self.der_layout = QVBoxLayout(self.panel_der)
        
        self.lbl_avatar = QLabel()
        self.lbl_avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_avatar.setFixedSize(120, 120)
        self.lbl_avatar.setStyleSheet("background: transparent;")
        
        self.lbl_nombre = QLabel("Seleccione un usuario")
        self.lbl_nombre.setObjectName("label-title")
        self.lbl_nombre.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.der_layout.addWidget(self.lbl_avatar, alignment=Qt.AlignmentFlag.AlignCenter)
        self.der_layout.addWidget(self.lbl_nombre, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.der_layout.addSpacing(20)
        lbl_permisos = QLabel("PERMISOS DE ÁREAS")
        lbl_permisos.setObjectName("label-section")
        self.der_layout.addWidget(lbl_permisos)

        # Contenedor para switches
        self.contenedor_switches = QWidget()
        self.layout_switches = QVBoxLayout(self.contenedor_switches)
        self.der_layout.addWidget(self.contenedor_switches)
        self.der_layout.addStretch()

        self.btn_guardar = QPushButton("Guardar Cambios")
        self.btn_guardar.setObjectName("btn-primary")
        self.btn_guardar.clicked.connect(self._guardar_cambios)
        self.btn_guardar.hide()
        self.der_layout.addWidget(self.btn_guardar)

        layout.addWidget(self.panel_der, stretch=7)
        self.usuario_actual = None
        self.switches_actuales = {}

    def cargar_datos(self):
        self.lista_usuarios.clear()
        self.usuarios_config = cargar_json(USUARIOS_CONFIG_FILE, {})
        self.areas_config = cargar_json(AREAS_CONFIG_FILE, {})
        
        for uname, udata in self.usuarios_config.items():
            item = QListWidgetItem(uname)
            # Intentar cargar avatar
            avatar_file = udata.get("avatar", "")
            if avatar_file:
                ruta = DIR_AVATARES / avatar_file
                if ruta.exists():
                    pix = QPixmap(str(ruta)).scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
                    item.setIcon(QIcon(pix))
            self.lista_usuarios.addItem(item)
            
        self.lbl_nombre.setText("Seleccione un usuario")
        self.lbl_avatar.clear()
        self.btn_guardar.hide()
        self._limpiar_switches()

    def _limpiar_switches(self):
        for i in reversed(range(self.layout_switches.count())): 
            widget = self.layout_switches.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        self.switches_actuales.clear()

    def _on_usuario_seleccionado(self, item):
        self.usuario_actual = item.text()
        self.lbl_nombre.setText(self.usuario_actual)
        self.btn_guardar.show()
        
        udata = self.usuarios_config.get(self.usuario_actual, {})
        avatar_file = udata.get("avatar", "")
        if avatar_file:
            ruta = DIR_AVATARES / avatar_file
            if ruta.exists():
                pix = QPixmap(str(ruta)).scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
                self.lbl_avatar.setPixmap(pix)
            else:
                self.lbl_avatar.setText("👤")
        else:
            self.lbl_avatar.setText("👤")
            self.lbl_avatar.setStyleSheet("font-size: 60px; background: transparent;")

        self._limpiar_switches()
        permisos_usuario = udata.get("permisos", [])
        
        for area_name, area_data in self.areas_config.items():
            desc = area_data.get("descripcion", "")
            switch = SwitchToggle(f"{area_name} - {desc}")
            switch.setChecked(area_name in permisos_usuario)
            self.layout_switches.addWidget(switch)
            self.switches_actuales[area_name] = switch

    def _guardar_cambios(self):
        if not self.usuario_actual:
            return
        
        pwd, ok = QInputDialog.getText(self, "Seguridad", "Ingrese la contraseña de administrador:", QLineEdit.EchoMode.Password)
        if ok and pwd == "admin":
            nuevos_permisos = []
            for area_name, switch in self.switches_actuales.items():
                if switch.isChecked():
                    nuevos_permisos.append(area_name)
                    
            self.usuarios_config[self.usuario_actual]["permisos"] = nuevos_permisos
            guardar_json(USUARIOS_CONFIG_FILE, self.usuarios_config)
            QMessageBox.information(self, "Éxito", f"Permisos actualizados para {self.usuario_actual}")
            self.cambios_guardados.emit()
        elif ok:
            QMessageBox.warning(self, "Denegado", "Contraseña incorrecta.")

# ---------------------------------------------------------------------------
# Widget: Dashboard Principal
# ---------------------------------------------------------------------------

class Dashboard(QWidget):
    """
    Layout de 3 columnas: Panel de Control (20%), Monitor (60%), Alertas (20%).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Columna izquierda (20%)
        self.panel_control = PanelControl()
        self.panel_control.setMinimumWidth(240)
        self.panel_control.setMaximumWidth(300)

        # Columna central (60%) -> Stacked Widget
        self.stack_central = QStackedWidget()
        self.panel_monitor = PanelMonitor()
        self.pantalla_permisos = PantallaPermisos()
        self.stack_central.addWidget(self.panel_monitor)
        self.stack_central.addWidget(self.pantalla_permisos)

        # Columna derecha (20%)
        self.panel_alertas = PanelAlertas()
        self.panel_alertas.setMinimumWidth(260)
        self.panel_alertas.setMaximumWidth(340)

        # Proporciones: 20% - 60% - 20%
        layout.addWidget(self.panel_control, stretch=2)
        layout.addWidget(self.stack_central, stretch=6)
        layout.addWidget(self.panel_alertas, stretch=2)


# ---------------------------------------------------------------------------
# Ventana Principal
# ---------------------------------------------------------------------------

class VentanaPrincipal(QMainWindow):
    """
    Ventana principal de la aplicación. Gestiona la transición entre
    la pantalla de login y el dashboard, arranca los hilos de IA y
    mantiene un ciclo de vida limpio de recursos.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("mini Argos — Sistema de Vigilancia Inteligente")

        # Contenedor central con stack de pantallas
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # Pantalla 0: Login
        self._pantalla_login = PantallaLogin(
            on_login_exitoso=self._mostrar_dashboard
        )
        self._stack.addWidget(self._pantalla_login)

        # Pantalla 1: Dashboard
        self._dashboard = Dashboard()
        self._stack.addWidget(self._dashboard)

        # Iniciar en login
        self._stack.setCurrentIndex(0)

        # Obtener referencias directas
        self.control = self._dashboard.panel_control
        self.monitor = self._dashboard.panel_monitor
        self.alertas = self._dashboard.panel_alertas

        # Conectar los eventos de los nuevos Pill Switches
        self.control.chk_cam1.stateChanged.connect(self._on_cam1_toggled)
        self.control.chk_cam2.stateChanged.connect(self._on_cam2_toggled)

        # Navegación del Dashboard
        self.control.btn_monitor.clicked.connect(self._nav_monitor)
        self.control.btn_restricciones.clicked.connect(self._nav_permisos)
        
        # Conectar reinicio táctico
        self._dashboard.pantalla_permisos.cambios_guardados.connect(self._reiniciar_camaras_tactico)
        
        # Conectar botón de registro al abrir VentanaRegistro
        self.control.btn_registro.clicked.connect(self._abrir_registro)

        # Motor de voz asíncrono (singleton para toda la sesión)
        self._cola_voz = queue.Queue()
        self._hilo_voz = HiloVoz(self._cola_voz)
        self._hilo_voz.start()

        # Referencia del hilo de IA y ventana de registro
        self.hilo_ia           = None
        self._ventana_registro = None

    def _mostrar_dashboard(self):
        """Transición de login a dashboard."""
        self._stack.setCurrentIndex(1)
        print("[APP] Dashboard cargado exitosamente.")

    def _nav_monitor(self):
        self._dashboard.stack_central.setCurrentIndex(0)
        self.control.btn_monitor.setChecked(True)
        self.control.btn_restricciones.setChecked(False)
        self.control.btn_areas.setChecked(False)

    def _nav_permisos(self):
        self._dashboard.stack_central.setCurrentIndex(1)
        self._dashboard.pantalla_permisos.cargar_datos()
        self.control.btn_monitor.setChecked(False)
        self.control.btn_restricciones.setChecked(True)
        self.control.btn_areas.setChecked(False)

    def _abrir_registro(self):
        """Abre la VentanaRegistro en el monitor secundario si está disponible."""
        self.control.btn_registro.clearFocus()
        # Si ya hay una instancia abierta, solo traerla al frente
        if self._ventana_registro and self._ventana_registro.isVisible():
            self._ventana_registro.raise_()
            self._ventana_registro.activateWindow()
            return

        self._ventana_registro = VentanaRegistro()
        # Conectar señal de éxito al slot de recarga en caliente del hilo IA y reinicio
        if self.hilo_ia:
            self._ventana_registro.registro_exitoso.connect(
                self.hilo_ia.recargar_embeddings
            )
        self._ventana_registro.registro_exitoso.connect(self._reiniciar_camaras_tactico)
        self._ventana_registro.show()

    def _reiniciar_camaras_tactico(self):
        print("[SISTEMA] Iniciando reinicio táctico de cámaras...")
        
        cam1_on = self.control.chk_cam1.isChecked()
        cam2_on = self.control.chk_cam2.isChecked()
        
        if self.hilo_ia:
            self.hilo_ia.recargar_configuracion()
            
        self.control.chk_cam1.setChecked(False)
        self.control.chk_cam2.setChecked(False)
        
        # Almacenar estados previos en la instancia para poder recuperarlos
        self._cam1_was_on = cam1_on
        self._cam2_was_on = cam2_on
        
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(5000, lambda: self._encender_camaras_previas())
            
        self._nav_monitor()

    def _encender_camaras_previas(self):
        if getattr(self, '_cam1_was_on', False):
            self.control.chk_cam1.setChecked(True)
        if getattr(self, '_cam2_was_on', False):
            self.control.chk_cam2.setChecked(True)

    def _on_cam1_toggled(self, state):
        """Manejador del switch para Cámara 1 — Inicia o detiene la IA."""
        cam1_active = (state == 2 or state == True)
        cam2_active = self.control.chk_cam2.isChecked()

        # Actualizar indicador Argos (Verde Neón si hay al menos una activa)
        self.control.actualizar_indicador_argos(cam1_active or cam2_active)

        # Adaptar layout del monitor
        self.monitor.actualizar_vistas(cam1_active, cam2_active)

        if cam1_active:
            # Leer URL validada de config.json
            url = cargar_config()
            if not url:
                print("[CAM] No hay URL RTMP guardada en config.json. Utilizando fallback por defecto.")
                # Fallback estándar si no está configurada
                url = "rtmp://localhost/live/stream"

            print(f"[CAM] Iniciando Cámara 1: {url}")

            # Detener hilo previo si existiera por seguridad
            if self.hilo_ia:
                self.hilo_ia.stop()
                self.hilo_ia = None

            # Lanzar Hilo IA de procesamiento en segundo plano
            self.hilo_ia = HiloProcesamientoIA(url, self._cola_voz)
            self.hilo_ia.signal_frame.connect(self.monitor.mostrar_frame)
            self.hilo_ia.signal_alerta.connect(self.alertas.agregar_alerta)
            self.hilo_ia.signal_stats.connect(self.control.actualizar_stats)
            self.hilo_ia.signal_error.connect(self._on_ia_error)
            self.hilo_ia.start()
        else:
            print("[CAM] Apagando Cámara 1.")
            if self.hilo_ia:
                self.hilo_ia.stop()
                self.hilo_ia = None

            # Limpieza del monitor e indicadores
            self.monitor.limpiar_monitor()
            self.control.actualizar_stats(0, 0, 0.0)

            # Silenciar inmediatamente alarmas activas
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

    def _on_cam2_toggled(self, state):
        """Manejador del switch para Cámara 2 — Dron inactivo."""
        cam1_active = self.control.chk_cam1.isChecked()
        cam2_active = (state == 2 or state == True)

        # Actualizar indicador Argos (Verde Neón si hay al menos una activa)
        self.control.actualizar_indicador_argos(cam1_active or cam2_active)

        # Adaptar layout del monitor (división horizontal)
        self.monitor.actualizar_vistas(cam1_active, cam2_active)
        print(f"[CAM] Cámara 2: {'ON' if cam2_active else 'OFF'}")

    def _on_ia_error(self, err_msg):
        """Callback para errores producidos en el hilo IA."""
        print(f"[ERROR IA] {err_msg}")
        self.control.chk_cam1.setChecked(False)

    def closeEvent(self, event):
        """Asegura el apagado correcto de todos los hilos al cerrar."""
        print("[APP] Cerrando aplicación de forma segura...")

        # Apagar inmediatamente cualquier sonido en bucle
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

        # Detener hilo IA
        if self.hilo_ia:
            print("[APP] Deteniendo Hilo IA secundario...")
            self.hilo_ia.stop()
            self.hilo_ia = None

        # Detener motor de voz (sentinel None + join)
        if hasattr(self, '_hilo_voz'):
            print("[APP] Deteniendo motor de voz TTS...")
            self._hilo_voz.detener()

        print("[APP] Recursos liberados con éxito.")
        event.accept()


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main():
    """Inicia la aplicación PyQt6 en pantalla completa."""
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    # Fuente global
    fuente = QFont("Segoe UI", 13)
    app.setFont(fuente)

    ventana = VentanaPrincipal()
    ventana.showFullScreen()

    print("[APP] mini Argos v1 — Interfaz iniciada.")
    print("[APP] Presione Alt+F4 para cerrar.")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
