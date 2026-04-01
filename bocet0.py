"""
Detección de rostros y emociones con DeepFace
Prototipo para la propuesta 3 del proyecto integrador:
usar expresiones faciales (emociones) como entrada al sistema educativo.
"""

import cv2
from deepface import DeepFace

# ============ INICIALIZAR CÁMARA ============

# 0 = cámara por defecto del sistema
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("No se pudo abrir la cámara.")
    raise SystemExit

print("Cámara abierta correctamente. Presiona 'q' para salir.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("No se pudo capturar el frame.")
        break

    # Opcional: redimensionar para mejor rendimiento en PCs lentas
    # frame = cv2.resize(frame, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)

    try:
        # DeepFace.analyze puede devolver una lista de resultados (uno por rostro)
        # enforce_detection=False evita que truene si hay frames sin rostros claros
        results = DeepFace.analyze(
            frame,
            actions=['emotion'],        # solo nos interesa emoción
            enforce_detection=False,    # no forzar detección (evita excepciones)
            silent=True                 # menos mensajes por consola
        )
    except Exception as e:
        print("Error en DeepFace.analyze:", e)
        continue

    # Normalizar formato de salida: lista de dicts
    if isinstance(results, list):
        faces_info = results
    else:
        faces_info = [results]

    emociones_en_frame = []

    for info in faces_info:
        # Región del rostro (bounding box)
        region = info.get('region', {})
        x = region.get('x', 0)
        y = region.get('y', 0)
        w = region.get('w', 0)
        h = region.get('h', 0)

        # Emoción dominante y distribución completa
        dominant_emotion = info.get('dominant_emotion', 'unknown')
        emotion_scores = info.get('emotion', {})

        emociones_en_frame.append((dominant_emotion, emotion_scores))

        # Dibujar solo si la caja tiene tamaño razonable
        if w > 0 and h > 0:
            # Recuadro verde alrededor del rostro
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Texto: emoción sobre la caja
            text = dominant_emotion
            cv2.putText(frame, text, (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Mostrar en consola lo que se detectó
    if emociones_en_frame:
        print("Emociones detectadas en este frame:")
        for i, (emo, scores) in enumerate(emociones_en_frame, start=1):
            print(f"  Rostro {i}: {emo}")
    else:
        print("No se detectaron emociones claras en este frame.")

    # Mostrar ventana con los resultados
    cv2.imshow("Rostros y emociones (DeepFace)", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Salida solicitada por el usuario (tecla 'q').")
        break

# ============ LIBERAR RECURSOS ============

video_capture.release()
cv2.destroyAllWindows()
