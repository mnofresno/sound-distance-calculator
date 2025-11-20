"""
Backend para detección de distancia a explosiones
Procesa video/audio y calcula distancia basada en diferencia temporal flash-boom
"""
import os
import tempfile
import json
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
import librosa
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from scipy import signal
from scipy.ndimage import uniform_filter1d

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sound Distance Calculator")

# CORS para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar directorio estático
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def detect_flash(video_path: str, fps: float) -> tuple[float, float]:
    """
    Detecta el instante del destello analizando cambios de brillo en frames.
    
    Returns:
        (t_flash_s, confidence): tiempo en segundos y estimación de error
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("No se pudo abrir el video")
    
    brightness_values = []
    frame_times = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Brillo medio del frame
        brightness = np.mean(gray)
        brightness_values.append(brightness)
        frame_times.append(frame_count / fps)
        frame_count += 1
    
    cap.release()
    
    if len(brightness_values) < 2:
        raise ValueError("Video demasiado corto")
    
    brightness_array = np.array(brightness_values)
    frame_times_array = np.array(frame_times)
    
    # Derivada temporal del brillo (diferencia entre frames consecutivos)
    brightness_diff = np.diff(brightness_array)
    

    # Suavizar para reducir ruido (usar ventana más pequeña para mejor resolución temporal)
    if len(brightness_diff) > 3:
        brightness_diff = uniform_filter1d(brightness_diff, size=3)
    
    # Filtrar valores negativos (solo aumentos de brillo)
    positive_diff = brightness_diff.copy()
    positive_diff[positive_diff < 0] = 0
    
    # Estadísticas para detectar picos significativos
    mean_diff = np.mean(positive_diff)
    std_diff = np.std(positive_diff)
    # Umbral más estricto: mean + 3*std (similar al boom pero un poco más permisivo)
    threshold = mean_diff + 3 * std_diff
    
    logger.info(f"Estadísticas de brillo: mean={mean_diff:.2f}, std={std_diff:.2f}, threshold={threshold:.2f}")
    
    # Buscar el PRIMER pico que supera el umbral (inicio del flash)
    peaks_above_threshold = np.where(positive_diff > threshold)[0]
    
    if len(peaks_above_threshold) == 0:
        # Si no hay picos claros, buscar el máximo pero con validación
        logger.warning("No se encontraron picos claros, usando máximo global")
        max_diff_idx = np.argmax(positive_diff)
        peak_value = positive_diff[max_diff_idx]
        
        # Validar que sea significativo
        if peak_value < mean_diff + 2 * std_diff:
            logger.warning(f"Pico de flash débil: {peak_value:.2f} vs umbral {mean_diff + 2*std_diff:.2f}")
        
        t_flash_s = frame_times_array[max_diff_idx]
        sigma_flash_detect = 0.1  # Error grande si no hay pico claro
    else:
        # PRIMER pico que supera el umbral (inicio del flash)
        first_peak_idx = peaks_above_threshold[0]
        peak_value = positive_diff[first_peak_idx]
        
        logger.info(f"Flash detectado en frame {first_peak_idx}, tiempo {frame_times_array[first_peak_idx]:.3f}s, valor={peak_value:.2f}")
        
        # El destello ocurre en el frame donde comienza el aumento
        t_flash_s = frame_times_array[first_peak_idx]
        
        # Calcular SNR para estimar error
        snr = (peak_value - mean_diff) / std_diff if std_diff > 0 else 1
        logger.info(f"SNR del flash: {snr:.2f}")
        
        # Error inversamente proporcional al SNR
        base_error = 1 / (2 * fps)  # Error mínimo (medio frame)
        sigma_flash_detect = base_error * max(1, 5 / snr)
    
    return float(t_flash_s), float(sigma_flash_detect)


def detect_boom(audio_path: str, sample_rate: Optional[int] = None) -> tuple[float, float, int]:
    """
    Detecta el instante del sonido analizando la envolvente del audio.
    
    Returns:
        (t_boom_s, confidence, actual_sample_rate): tiempo en segundos, estimación de error y sample rate real
    """
    # Cargar audio
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    
    # Usar el sample rate del audio cargado
    actual_sample_rate = sr
    
    # Rectificar (valor absoluto)
    audio_rectified = np.abs(audio)
    
    # Envolvente usando ventana móvil (promedio móvil) - ventana más pequeña para mejor resolución
    window_size = int(0.005 * actual_sample_rate)  # 5ms (reducido de 10ms)
    if window_size < 1:
        window_size = 1
    
    envelope = uniform_filter1d(audio_rectified, size=window_size)
    
    # Calcular la derivada temporal de la envolvente (cambio de amplitud)
    envelope_diff = np.diff(envelope)
    
    # Filtrar solo aumentos (valores positivos)
    positive_diff = envelope_diff.copy()
    positive_diff[positive_diff < 0] = 0
    
    # Estadísticas para detectar aumentos significativos
    mean_diff = np.mean(positive_diff)
    std_diff = np.std(positive_diff)
    # Umbral: mean + 3*std (similar al flash)
    threshold = mean_diff + 3 * std_diff
    
    logger.info(f"Estadísticas de audio: mean_diff={mean_diff:.6f}, std_diff={std_diff:.6f}, threshold={threshold:.6f}")
    
    # Buscar el PRIMER aumento significativo (inicio del boom)
    increases_above_threshold = np.where(positive_diff > threshold)[0]
    
    if len(increases_above_threshold) == 0:
        # Si no hay aumentos claros, usar el máximo de la envolvente
        max_idx = np.argmax(envelope)
        t_boom_s = max_idx / actual_sample_rate
        logger.warning(f"No se encontraron aumentos claros, usando máximo de envolvente en {t_boom_s:.3f}s")
        sigma_boom_detect = 0.1  # 100ms de error estimado
    else:
        # PRIMER aumento que supera el umbral (inicio del boom)
        first_increase_idx = increases_above_threshold[0]
        t_boom_s = first_increase_idx / actual_sample_rate
        
        peak_value = positive_diff[first_increase_idx]
        snr = (peak_value - mean_diff) / std_diff if std_diff > 0 else 1
        
        logger.info(f"Boom detectado en muestra {first_increase_idx}, tiempo {t_boom_s:.3f}s, valor={peak_value:.6f}, SNR={snr:.2f}")
        
        # Error inversamente proporcional al SNR
        base_error = 1 / actual_sample_rate  # Error mínimo (1 muestra)
        sigma_boom_detect = base_error * max(1, 5 / snr)
    
    return float(t_boom_s), float(sigma_boom_detect), int(actual_sample_rate)


def calculate_distance(
    t_flash_s: float,
    t_boom_s: float,
    fps: float,
    sample_rate: int,
    temperature_celsius: float = 20.0,
    sigma_flash_detect: float = 0.0,
    sigma_boom_detect: float = 0.0,
    user_adjusted: bool = False,
    strict_validation: bool = True
) -> dict:
    """
    Calcula distancia y propaga errores.
    
    Args:
        strict_validation: Si False, permite delta_t fuera de rango pero marca como baja confianza
    """
    # Validación delta_t
    delta_t = t_boom_s - t_flash_s
    
    if strict_validation and (delta_t < 0.05 or delta_t > 10.0):
        raise ValueError(f"delta_t ({delta_t:.3f}s) fuera del rango válido [0.05, 10.0]s")
    
    # Velocidad del sonido
    speed_of_sound = 331.0 + 0.6 * temperature_celsius
    
    # Distancia
    distance_m = speed_of_sound * delta_t
    
    # Errores por discretización
    sigma_frame = 1 / (2 * fps)
    sigma_audio = 1 / sample_rate
    
    # Si el usuario ajustó manualmente, reducir errores de detección
    if user_adjusted:
        sigma_flash_detect = sigma_frame  # Solo error de discretización
        sigma_boom_detect = sigma_audio
    
    # Propagación de errores
    sigma_dt = np.sqrt(
        sigma_frame ** 2 +
        sigma_audio ** 2 +
        sigma_flash_detect ** 2 +
        sigma_boom_detect ** 2
    )
    
    sigma_d = sigma_dt * speed_of_sound
    
    # Confianza de detección automática
    if user_adjusted:
        confidence = "user_verified"
    else:
        total_auto_error = np.sqrt(sigma_flash_detect ** 2 + sigma_boom_detect ** 2)
        if total_auto_error < 0.01:
            confidence = "high"
        elif total_auto_error < 0.05:
            confidence = "medium"
        else:
            confidence = "low"
    
    return {
        "t_flash_s": round(t_flash_s, 6),
        "t_boom_s": round(t_boom_s, 6),
        "delta_t_s": round(delta_t, 6),
        "distance_m": round(distance_m, 2),
        "error_m": round(sigma_d, 2),
        "fps": round(fps, 2),
        "sample_rate": int(sample_rate),
        "speed_of_sound_m_s": round(speed_of_sound, 2),
        "auto_detection_confidence": confidence,
        "user_adjusted": user_adjusted
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    """Sirve la interfaz HTML"""
    html_path = static_dir / "index.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Error: index.html no encontrado</h1>"


@app.post("/api/process")
async def process_video(
    file: UploadFile = File(...),
    temperature: Optional[float] = Form(None)
):
    """
    Procesa video y detecta automáticamente flash y boom.
    """
    audio_path = None
    video_path = None
    
    try:
        # Validar que sea un archivo de video
        if not file.filename:
            logger.error("No se proporcionó un archivo")
            raise HTTPException(status_code=400, detail="No se proporcionó un archivo")
        
        logger.info(f"Procesando archivo: {file.filename}, tipo: {file.content_type}")
        
        # Crear archivo temporal con extensión correcta
        file_ext = os.path.splitext(file.filename)[1].lower()
        if not file_ext:
            file_ext = ".mp4"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_video:
            video_path = tmp_video.name
            content = await file.read()
            if len(content) == 0:
                logger.error("El archivo está vacío")
                raise HTTPException(status_code=400, detail="El archivo está vacío")
            tmp_video.write(content)
            logger.info(f"Archivo temporal creado: {video_path}, tamaño: {len(content)} bytes")
        
        # Obtener propiedades del video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"No se pudo abrir el video: {video_path}")
            raise HTTPException(status_code=400, detail="No se pudo abrir el video. Verifique que sea un formato de video válido.")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        logger.info(f"Video propiedades: FPS={fps}, frames={frame_count}, resolución={width}x{height}")
        
        if fps <= 0 or fps > 1000:
            logger.error(f"FPS inválido: {fps}")
            raise HTTPException(status_code=400, detail=f"FPS inválido en el video: {fps}")
        
        if frame_count <= 0:
            logger.error(f"Video sin frames: {frame_count}")
            raise HTTPException(status_code=400, detail="El video no tiene frames")
        
        duration = frame_count / fps if fps > 0 else 0
        logger.info(f"Duración del video: {duration:.2f}s")
        
        # Extraer audio
        audio_path = video_path.replace(file_ext, ".wav")
        try:
            # Usar librosa directamente del video (maneja mejor los formatos)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                audio, sample_rate = librosa.load(video_path, sr=None, mono=True)
            
            if len(audio) == 0:
                raise HTTPException(status_code=400, detail="El video no contiene audio")
            
            # Guardar audio temporal para detección
            sf.write(audio_path, audio, sample_rate)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error extrayendo audio del video: {str(e)}")
        
        # Detección automática
        try:
            logger.info("Iniciando detección de flash...")
            t_flash_s, sigma_flash_detect = detect_flash(video_path, fps)
            logger.info(f"Flash detectado en: {t_flash_s:.6f}s (error: {sigma_flash_detect:.6f}s)")
        except Exception as e:
            logger.error(f"Error detectando flash: {str(e)}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Error detectando flash: {str(e)}")
        
        try:
            logger.info("Iniciando detección de boom...")
            t_boom_s, sigma_boom_detect, actual_sample_rate = detect_boom(audio_path)
            logger.info(f"Boom detectado en: {t_boom_s:.6f}s (error: {sigma_boom_detect:.6f}s), sample_rate: {actual_sample_rate}")
        except Exception as e:
            logger.error(f"Error detectando boom: {str(e)}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Error detectando boom: {str(e)}")
        
        # Usar el sample rate real del audio
        sample_rate = actual_sample_rate
        
        # Validar y ajustar tiempos si están fuera de rango
        if t_flash_s < 0:
            logger.warning(f"Tiempo de flash negativo ({t_flash_s:.3f}s), ajustando a 0")
            t_flash_s = 0.0
        elif t_flash_s > duration:
            logger.warning(f"Tiempo de flash ({t_flash_s:.3f}s) mayor que duración ({duration:.2f}s), ajustando")
            t_flash_s = duration * 0.9  # Ajustar al 90% de la duración
        
        if t_boom_s < 0:
            logger.warning(f"Tiempo de boom negativo ({t_boom_s:.3f}s), ajustando a 0")
            t_boom_s = 0.0
        elif t_boom_s > duration:
            logger.warning(f"Tiempo de boom ({t_boom_s:.3f}s) mayor que duración ({duration:.2f}s), ajustando")
            t_boom_s = duration * 0.95  # Ajustar al 95% de la duración
        
        # Validar que boom sea después de flash (físicamente debe ser así)
        if t_boom_s < t_flash_s:
            logger.warning(f"Boom ({t_boom_s:.3f}s) antes que flash ({t_flash_s:.3f}s), esto es inusual pero permitido")
        
        # Temperatura (CNPT si no se especifica)
        if temperature is None:
            temperature = 20.0  # CNPT
        
        # Cálculo inicial - permitir delta_t fuera de rango pero con advertencia
        delta_t = t_boom_s - t_flash_s
        if delta_t < 0.05 or delta_t > 10.0:
            # Permitir el cálculo pero marcar como baja confianza
            pass
        
        try:
            result = calculate_distance(
                t_flash_s=t_flash_s,
                t_boom_s=t_boom_s,
                fps=fps,
                sample_rate=sample_rate,
                temperature_celsius=temperature,
                sigma_flash_detect=sigma_flash_detect,
                sigma_boom_detect=sigma_boom_detect,
                user_adjusted=False,
                strict_validation=False  # Permitir cálculos fuera de rango con advertencia
            )
        except ValueError as e:
            # Si el error es por delta_t fuera de rango, calcular de todos modos
            if "delta_t" in str(e).lower():
                # Forzar cálculo sin validación estricta
                speed_of_sound = 331.0 + 0.6 * temperature
                distance_m = speed_of_sound * delta_t
                sigma_frame = 1 / (2 * fps)
                sigma_audio = 1 / sample_rate
                sigma_dt = np.sqrt(
                    sigma_frame ** 2 +
                    sigma_audio ** 2 +
                    sigma_flash_detect ** 2 +
                    sigma_boom_detect ** 2
                )
                sigma_d = sigma_dt * speed_of_sound
                
                result = {
                    "t_flash_s": round(t_flash_s, 6),
                    "t_boom_s": round(t_boom_s, 6),
                    "delta_t_s": round(delta_t, 6),
                    "distance_m": round(distance_m, 2),
                    "error_m": round(sigma_d, 2),
                    "fps": round(fps, 2),
                    "sample_rate": int(sample_rate),
                    "speed_of_sound_m_s": round(speed_of_sound, 2),
                    "auto_detection_confidence": "low",
                    "user_adjusted": False,
                    "warning": f"delta_t ({delta_t:.3f}s) fuera del rango recomendado [0.05, 10.0]s"
                }
            else:
                raise HTTPException(status_code=400, detail=str(e))
        
        # Agregar información adicional para la UI
        result["duration"] = round(duration, 2)
        result["video_url"] = f"/static/uploads/{file.filename}"
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"Error procesando video: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)
    finally:
        # Limpiar archivos temporales
        try:
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)
        except Exception:
            pass


@app.post("/api/calculate")
async def calculate(
    t_flash_s: float = Form(...),
    t_boom_s: float = Form(...),
    fps: float = Form(...),
    sample_rate: int = Form(...),
    temperature: Optional[float] = Form(None),
    user_adjusted: bool = Form(True)
):
    """
    Recalcula distancia con valores ajustados por el usuario.
    """
    try:
        if temperature is None:
            temperature = 20.0  # CNPT
        
        result = calculate_distance(
            t_flash_s=t_flash_s,
            t_boom_s=t_boom_s,
            fps=fps,
            sample_rate=sample_rate,
            temperature_celsius=temperature,
            user_adjusted=user_adjusted
        )
        
        return JSONResponse(content=result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculando: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)

