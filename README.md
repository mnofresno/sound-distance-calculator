# Calculadora de Distancia a Explosiones

Herramienta para calcular la distancia a una explosión basándose en la diferencia temporal entre el destello visual y el sonido.

## Características

- **Detección automática** de destello (flash) y sonido (boom)
- **Interfaz interactiva** con reproductor de video y marcadores editables
- **Cálculo de distancia** con propagación de errores
- **Validaciones** de rangos y consistencia
- **Visualización de audio** en la zona del boom

## Instalación

1. Instalar dependencias:

```bash
pip install -r requirements.txt
```

2. Ejecutar el servidor:

```bash
python app.py
```

O con uvicorn directamente:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

3. Abrir en el navegador:

```
http://localhost:8000
```

## Uso

1. Cargar un video que contenga una explosión (con audio)
2. El sistema detectará automáticamente:
   - El instante del destello (análisis de brillo de frames)
   - El instante del sonido (análisis de envolvente de audio)
3. Ajustar manualmente los marcadores en la línea de tiempo si es necesario
4. Especificar temperatura (opcional, por defecto usa CNPT: 20°C)
5. Ver resultados con distancia y error estimado

## Algoritmos

### Detección de Flash

- Conversión de frames a escala de grises
- Cálculo de brillo medio por frame
- Derivada temporal del brillo
- Selección del frame con máxima derivada

### Detección de Boom

- Audio convertido a mono
- Rectificación (valor absoluto)
- Envolvente con ventana móvil
- Primer pico que supera `mean + 4*std`

### Cálculo de Distancia

```
v = 331 + 0.6 * temperatura_celsius
distance = v * (t_boom - t_flash)
```

### Propagación de Errores

- Error por discretización de video: `1/(2*fps)`
- Error por discretización de audio: `1/sample_rate`
- Error por detección automática (estimado según SNR y forma del pico)
- Error total: propagación cuadrática

## API

### POST `/api/process`

Procesa un video y detecta automáticamente flash y boom.

**Parámetros:**
- `file`: Archivo de video
- `temperature` (opcional): Temperatura en °C

**Respuesta:**
```json
{
  "t_flash_s": 1.234567,
  "t_boom_s": 1.456789,
  "delta_t_s": 0.222222,
  "distance_m": 75.50,
  "error_m": 2.30,
  "fps": 30.0,
  "sample_rate": 44100,
  "speed_of_sound_m_s": 343.0,
  "auto_detection_confidence": "high",
  "user_adjusted": false
}
```

### POST `/api/calculate`

Recalcula distancia con valores ajustados por el usuario.

**Parámetros:**
- `t_flash_s`: Tiempo del flash en segundos
- `t_boom_s`: Tiempo del boom en segundos
- `fps`: FPS del video
- `sample_rate`: Sample rate del audio
- `temperature` (opcional): Temperatura en °C
- `user_adjusted`: true/false

## Validaciones

- `delta_t` debe estar entre 0.05s y 10.0s
- Si está fuera de rango, se muestra advertencia pero se permite el cálculo

## Notas Técnicas

- Los frames se procesan en memoria (no se guardan en disco)
- El audio se extrae usando librosa
- La detección automática puede requerir ajuste manual en casos de bajo SNR
- La confianza de detección se estima según el error total

