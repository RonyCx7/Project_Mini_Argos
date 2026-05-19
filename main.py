"""
Project mini-Argos v1 — Fases 1, 2 y 3.2: UI + Stream + Detección con YOLO
===========================================================================
Módulo principal del proyecto. Gestiona la configuración RTMP mediante una
ventana tkinter y, una vez validada la URL, abre el stream y detecta personas
en tiempo real usando YOLOv8 acelerado por GPU (CUDA).

Flujo completo:
    Fase 1 — UI de configuración:
        1. Carga la última URL RTMP desde `config.json` (si existe).
        2. Muestra la ventana de configuración al usuario.
        3. Valida la URL ingresada al presionar "Conectar".
        4. Si es válida → guarda en `config.json` y destruye la ventana.
        5. Si es inválida → muestra un messagebox de error sin cerrar la ventana.

    Fase 2 — Captura del stream (con multithreading):
        6. Recupera la URL validada desde `config.json`.
        7. Abre el stream RTMP con cv2.VideoCapture.
        8. Un hilo en segundo plano (LectorStream) drena el buffer RTMP
           continuamente, conservando siempre solo el último frame para
           eliminar la acumulación de latencia.

    Fase 3.2 — Detección de personas (YOLOv8 + CUDA):
        9. Carga el modelo YOLOv8n (se descarga automáticamente la primera vez).
       10. El hilo principal toma el último frame del LectorStream, ejecuta
           la inferencia YOLO filtrando solo la clase 'person' (0), dibuja
           bounding boxes y superpone el estado en pantalla.
       11. El usuario puede presionar 'q' para cerrar limpiamente.
"""

import json
import sys
import threading
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Constantes de configuración
# ---------------------------------------------------------------------------

CONFIG_FILE = Path(__file__).parent / "config.json"
CONFIG_KEY  = "rtmp_url"

# Paleta de colores — estilo oscuro moderno
CLR_BG          = "#1E1E2E"   # Fondo principal (navy oscuro)
CLR_SURFACE     = "#2A2A3E"   # Superficie de tarjeta
CLR_ACCENT      = "#7C6AF7"   # Morado vibrante (botón activo)
CLR_ACCENT_HOV  = "#9A8AFF"   # Hover del botón
CLR_TEXT        = "#CDD6F4"   # Texto principal
CLR_SUBTEXT     = "#A6ADC8"   # Texto secundario / placeholder
CLR_ENTRY_BG    = "#313244"   # Fondo del Entry
CLR_BORDER      = "#45475A"   # Borde sutil
CLR_ERROR       = "#F38BA8"   # Rojo suave para errores


# ---------------------------------------------------------------------------
# Capa de persistencia (config.json)
# ---------------------------------------------------------------------------

def cargar_config() -> str:
    """
    Lee `config.json` y retorna la URL RTMP almacenada.

    Returns:
        str: La URL guardada, o cadena vacía si el archivo no existe
             o el campo no está presente.
    """
    if CONFIG_FILE.exists():
        try:
            with CONFIG_FILE.open("r", encoding="utf-8") as f:
                datos = json.load(f)
            return datos.get(CONFIG_KEY, "")
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[ADVERTENCIA] No se pudo leer {CONFIG_FILE.name}: {exc}")
    return ""


def guardar_config(url: str) -> None:
    """
    Escribe (o sobreescribe) `config.json` con la URL RTMP proporcionada.

    Args:
        url (str): Dirección RTMP validada que se desea persistir.
    """
    datos = {CONFIG_KEY: url}
    try:
        with CONFIG_FILE.open("w", encoding="utf-8") as f:
            json.dump(datos, f, indent=4, ensure_ascii=False)
        print(f"[CONFIG] URL guardada en '{CONFIG_FILE.name}'.")
    except OSError as exc:
        print(f"[ERROR] No se pudo guardar la configuración: {exc}")


# ---------------------------------------------------------------------------
# Capa de validación
# ---------------------------------------------------------------------------

def validar_rtmp(url: str) -> bool:
    """
    Valida que la cadena ingresada sea una URL RTMP con estructura mínima.

    Criterios:
        - No puede estar vacía.
        - Debe comenzar con el esquema 'rtmp://'.
        - Debe tener al menos un carácter después del prefijo.

    Args:
        url (str): Cadena a validar.

    Returns:
        bool: True si pasa la validación, False en caso contrario.
    """
    url = url.strip()
    return bool(url) and url.lower().startswith("rtmp://") and len(url) > len("rtmp://")


# ---------------------------------------------------------------------------
# Capa de UI (tkinter)
# ---------------------------------------------------------------------------

class VentanaConfig:
    """
    Ventana de configuración RTMP con estilo oscuro moderno.

    La ventana es modal (bloquea el hilo principal hasta que el usuario
    interactúa), por lo que se usa `root.mainloop()` de forma estándar.
    """

    def __init__(self, root: tk.Tk, url_inicial: str = "") -> None:
        self.root = root
        self._construir_ventana(url_inicial)

    # ------------------------------------------------------------------
    # Construcción de la interfaz
    # ------------------------------------------------------------------

    def _construir_ventana(self, url_inicial: str) -> None:
        """Configura la ventana principal y todos sus widgets."""

        raiz = self.root

        # ── Propiedades de la ventana ──────────────────────────────────
        raiz.title("mini Argos — Configuración")
        raiz.configure(bg=CLR_BG)
        raiz.resizable(False, False)

        # Centrar en pantalla
        ancho, alto = 480, 240
        x = (raiz.winfo_screenwidth()  - ancho) // 2
        y = (raiz.winfo_screenheight() - alto)  // 2
        raiz.geometry(f"{ancho}x{alto}+{x}+{y}")

        # ── Contenedor tipo "tarjeta" ──────────────────────────────────
        tarjeta = tk.Frame(raiz, bg=CLR_SURFACE, padx=30, pady=24,
                           relief="flat")
        tarjeta.pack(fill="both", expand=True, padx=16, pady=16)

        # ── Título de la aplicación ────────────────────────────────────
        tk.Label(
            tarjeta,
            text="🎥  mini Argos",
            font=("Segoe UI", 13, "bold"),
            bg=CLR_SURFACE,
            fg=CLR_ACCENT,
        ).pack(anchor="w")

        # ── Separador visual ───────────────────────────────────────────
        tk.Frame(tarjeta, height=1, bg=CLR_BORDER).pack(fill="x", pady=(6, 14))

        # ── Etiqueta instructiva ───────────────────────────────────────
        tk.Label(
            tarjeta,
            text="Ingrese la dirección del stream RTMP:",
            font=("Segoe UI", 10),
            bg=CLR_SURFACE,
            fg=CLR_SUBTEXT,
        ).pack(anchor="w")

        # ── Campo de entrada ───────────────────────────────────────────
        self.entry_url = tk.Entry(
            tarjeta,
            font=("Consolas", 11),
            bg=CLR_ENTRY_BG,
            fg=CLR_TEXT,
            insertbackground=CLR_TEXT,      # color del cursor de texto
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightcolor=CLR_ACCENT,
            highlightbackground=CLR_BORDER,
        )
        self.entry_url.pack(fill="x", ipady=8, pady=(6, 16))

        # Insertar URL precargada (si existe)
        if url_inicial:
            self.entry_url.insert(0, url_inicial)
            self.entry_url.icursor("end")  # posicionar cursor al final

        # ── Botón "Conectar" ───────────────────────────────────────────
        self.btn_conectar = tk.Button(
            tarjeta,
            text="  Conectar  ",
            font=("Segoe UI", 10, "bold"),
            bg=CLR_ACCENT,
            fg="#FFFFFF",
            activebackground=CLR_ACCENT_HOV,
            activeforeground="#FFFFFF",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=12,
            pady=8,
            command=self._on_conectar,
        )
        self.btn_conectar.pack(anchor="e")

        # Efectos hover del botón
        self.btn_conectar.bind("<Enter>", lambda _: self.btn_conectar.config(bg=CLR_ACCENT_HOV))
        self.btn_conectar.bind("<Leave>", lambda _: self.btn_conectar.config(bg=CLR_ACCENT))

        # Atajo de teclado: Enter dispara el botón
        raiz.bind("<Return>", lambda _: self._on_conectar())

    # ------------------------------------------------------------------
    # Lógica del botón "Conectar"
    # ------------------------------------------------------------------

    def _on_conectar(self) -> None:
        """
        Manejador del evento 'Conectar'.

        - Valida la URL ingresada.
        - Si es inválida:  muestra un messagebox de error.
        - Si es válida:    persiste la URL, destruye la ventana e imprime
                           la confirmación en consola.
        """
        url = self.entry_url.get().strip()

        if not validar_rtmp(url):
            messagebox.showerror(
                title="URL inválida",
                message=(
                    "La dirección ingresada no es válida.\n\n"
                    "Asegúrese de que comience con:\n"
                    "  rtmp://servidor/ruta"
                ),
                parent=self.root,
            )
            # Resaltar el borde del Entry en rojo para retroalimentación visual
            self.entry_url.config(highlightcolor=CLR_ERROR, highlightbackground=CLR_ERROR)
            self.entry_url.focus_set()
            return

        # Restablecer el borde en caso de corrección
        self.entry_url.config(highlightcolor=CLR_ACCENT, highlightbackground=CLR_BORDER)

        # Persistir y salir
        guardar_config(url)
        self.root.destroy()
        print(f"Iniciando conexión a: {url}")


# ---------------------------------------------------------------------------
# Fase 3.2: Captura RTMP (Multithreading) + Detección con YOLO (GPU)
# ---------------------------------------------------------------------------

# Constantes de visualización — se definen a nivel de módulo para no
# recrear objetos en cada iteración del bucle de captura.
_FUENTE        = cv2.FONT_HERSHEY_SIMPLEX
_ESCALA_TEXTO  = 0.75
_GROSOR_TEXTO  = 2
_COLOR_VERDE   = (0, 220, 80)    # BGR — persona detectada
_COLOR_ROJO    = (50, 50, 235)   # BGR — sin persona
_COLOR_BOX     = (124, 106, 247) # BGR — bounding box (morado del tema)
_POS_TEXTO     = (14, 38)        # Margen superior izquierdo del overlay


class LectorStream:
    """
    Lector de stream RTMP en un hilo en segundo plano.

    Problema que resuelve:
        OpenCV almacena internamente un buffer de frames del stream RTMP.
        Si el hilo principal no los consume lo suficientemente rápido
        (porque está ocupado con la inferencia de YOLO), el buffer se llena
        y los frames mostrados tienen varios segundos de retraso.

    Solución:
        Un hilo dedicado ejecuta `cap.read()` en un bucle continuo, drenando
        el buffer lo más rápido posible. Solo conserva el *último* frame
        leído en `self.frame`, que el hilo principal consulta cuando esté
        listo para procesar.

    Uso:
        lector = LectorStream(url)
        lector.iniciar()
        ...
        frame = lector.leer()   # siempre el frame más reciente
        ...
        lector.detener()
    """

    def __init__(self, url: str) -> None:
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self._detenido = False
        self._lock = threading.Lock()
        self._hilo = None

    @property
    def esta_abierto(self) -> bool:
        """True si el VideoCapture subyacente se abrió correctamente."""
        return self.cap.isOpened()

    def iniciar(self) -> None:
        """
        Arranca el hilo de lectura en segundo plano.

        El hilo es daemon=True para que no impida la salida del proceso
        si el hilo principal termina inesperadamente.
        """
        self._hilo = threading.Thread(target=self._bucle_lectura, daemon=True)
        self._hilo.start()
        print("[HILO] Lector de stream iniciado en segundo plano.")

    def _bucle_lectura(self) -> None:
        """
        Bucle interno del hilo: lee frames sin parar y conserva solo el último.

        Cada `cap.read()` drena un frame del buffer interno de OpenCV.
        El `Lock` protege la escritura de `self.frame` contra lecturas
        concurrentes del hilo principal.
        """
        while not self._detenido:
            ret, frame = self.cap.read()
            if not ret:
                # Stream interrumpido o finalizado
                self._detenido = True
                break
            with self._lock:
                self.frame = frame

    def leer(self):
        """
        Obtiene el último frame disponible (thread-safe).

        Returns:
            numpy.ndarray | None: El frame BGR más reciente, o None si
                                  aún no se ha leído ningún frame.
        """
        with self._lock:
            return self.frame

    def detener(self) -> None:
        """
        Señala al hilo que se detenga, espera a que termine y libera
        el objeto VideoCapture.
        """
        self._detenido = True
        if self._hilo is not None:
            self._hilo.join(timeout=3.0)
        self.cap.release()
        print("[HILO] Lector de stream detenido y recurso liberado.")


def iniciar_stream(url: str) -> None:
    """
    Abre el stream RTMP, detecta personas con YOLOv8 (GPU) y muestra
    los frames anotados en tiempo real mediante OpenCV.

    Arquitectura de hilos:
        - Hilo secundario (LectorStream): drena el buffer RTMP, conserva
          solo el último frame → latencia de buffer = 0.
        - Hilo principal: toma el último frame, ejecuta inferencia YOLO,
          dibuja bounding boxes + texto de estado, muestra resultado.

    Pipeline por frame (hilo principal):
        1. lector.leer()                → último frame BGR (sin latencia)
        2. modelo(frame, classes=[0])   → inferencia YOLO (GPU/CUDA)
        3. Extraer bounding boxes       → coordenadas (x1, y1, x2, y2)
        4. Anotación sobre frame BGR    → rectángulos + texto de estado
        5. cv2.imshow()                 → frame anotado en BGR

    El bucle termina cuando:
        - El usuario presiona 'q'.
        - El LectorStream señala que el stream se perdió.

    Al salir se detiene el hilo lector y se destruyen las ventanas OpenCV.

    Args:
        url (str): Dirección RTMP validada y persistida por la Fase 1.
    """
    # ── Cargar modelo YOLO ─────────────────────────────────────────────
    # yolov8n.pt = modelo "nano" (el más rápido). Se descarga automática-
    # mente la primera vez. Ultralytics detectará CUDA si está disponible.
    print("[YOLO] Cargando modelo YOLOv8n...")
    modelo = YOLO("yolov8n.pt")
    print("[YOLO] Modelo cargado correctamente.")

    # ── Iniciar lector de stream con hilo en segundo plano ─────────────
    print(f"[STREAM] Conectando a: {url}")
    lector = LectorStream(url)

    if not lector.esta_abierto:
        print(
            "[ERROR] No se pudo abrir el stream.\n"
            "        Verifique que el servidor RTMP esté activo y la URL sea correcta."
        )
        sys.exit(1)

    lector.iniciar()
    print("[STREAM] Conexión establecida. Presione 'q' para salir.")

    nombre_ventana = "mini Argos - Stream en Vivo"

    # ── Bucle principal de detección (hilo principal) ──────────────────
    while True:
        frame = lector.leer()

        # El lector aún no tiene un frame listo (arranque) o el stream murió
        if frame is None:
            if lector._detenido:
                print("[STREAM] El stream se ha interrumpido.")
                break
            # Esperar brevemente a que el hilo lector obtenga el primer frame
            cv2.waitKey(30)
            continue

        # 1. Inferencia YOLO — solo clase 0 (person), sin verbose.
        #    Ultralytics usa CUDA automáticamente si torch.cuda.is_available().
        resultados = modelo(frame, classes=[0], verbose=False)

        # 2. Extraer bounding boxes del primer (y único) resultado
        cajas = resultados[0].boxes

        # 3. Anotar el frame BGR original
        if len(cajas) > 0:
            for caja in cajas:
                # xyxy devuelve tensor [x1, y1, x2, y2] en píxeles absolutos
                x1, y1, x2, y2 = caja.xyxy[0].int().tolist()
                cv2.rectangle(
                    frame,
                    (x1, y1), (x2, y2),
                    _COLOR_BOX,
                    thickness=2,
                )

            # Overlay de estado — texto verde
            estado_texto = f"ESTADO: {len(cajas)} Persona(s) detectada(s)"
            cv2.putText(
                frame, estado_texto, _POS_TEXTO,
                _FUENTE, _ESCALA_TEXTO, _COLOR_VERDE, _GROSOR_TEXTO,
                cv2.LINE_AA,
            )
        else:
            # Overlay de estado — texto rojo
            cv2.putText(
                frame, "ESTADO: No hay personas", _POS_TEXTO,
                _FUENTE, _ESCALA_TEXTO, _COLOR_ROJO, _GROSOR_TEXTO,
                cv2.LINE_AA,
            )

        # 4. Mostrar el frame anotado (BGR, correcto para OpenCV)
        cv2.imshow(nombre_ventana, frame)

        # Salir con 'q' (& 0xFF normaliza el código en sistemas de 64-bit)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[STREAM] Tecla 'q' detectada. Cerrando stream...")
            break

    # ── Limpieza de recursos ───────────────────────────────────────────
    lector.detener()
    cv2.destroyAllWindows()
    print("[STREAM] Recursos liberados. Programa finalizado.")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Punto de entrada principal del programa (Fases 1, 2 y 3.2).

    Secuencia:
        1. Carga la configuración previa y lanza la UI de tkinter (Fase 1).
        2. Una vez cerrada la UI con éxito, recupera la URL validada
           leyendo nuevamente `config.json` (fuente de verdad compartida).
        3. Si existe una URL válida, inicia el stream con detección de
           personas vía YOLO + multithreading (Fases 2 + 3.2).
        4. Si la ventana fue cerrada sin guardar (e.g. clic en la X),
           la URL estará vacía y el programa termina sin iniciar el stream.
    """
    # ── Fase 1: UI de configuración ────────────────────────────────────
    url_guardada = cargar_config()

    root = tk.Tk()
    VentanaConfig(root, url_inicial=url_guardada)
    root.mainloop()   # Bloquea hasta que el usuario destruya la ventana

    # ── Fases 2 + 3.2: Stream con YOLO + multithreading ───────────────
    # Releer config.json como fuente de verdad: garantiza que usamos la
    # URL recién guardada incluso si el usuario la modificó en la UI.
    url_activa = cargar_config()

    if not url_activa:
        # El usuario cerró la ventana con la 'X' sin presionar "Conectar"
        print("[INFO] No se proporcionó ninguna URL RTMP. Programa cerrado.")
        sys.exit(0)

    iniciar_stream(url_activa)


if __name__ == "__main__":
    main()
