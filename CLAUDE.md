# Reconocimiento de señas aisladas en LSC (LatAm AI Challenge 2026)

Clasificación de videos de señas aisladas (dataset LSC5, 5 clases) para el reto binacional UNAB–UdeA.
Specs en `specs/` (requirements, design, tasks). Leerlos antes de tocar código.

## Reglas del reto (no negociables)
- Solo el dataset oficial LSC5. Datos externos se declaran y quedan a criterio del comité.
- No buscar ni inferir las etiquetas del test, no usar nombres originales ni metadatos, no intercambiar predicciones con otros equipos.
- Los datos NO se redistribuyen ni se suben a ningún repo; no intentar identificar a las personas grabadas.
- Los organizadores entregan 3 participantes etiquetados (train/val) y el test es el 4.º participante, sin etiquetas.
- Validación SIEMPRE por firmante: LOSO sobre los 3 participantes públicos (3 folds). Nunca mezclar firmantes entre train y val.
- Las 3 vistas de una misma ejecución van juntas en el mismo lado de cualquier split adicional.
- Estadísticas de normalización: solo del train del fold. El test solo se usa para inferencia (ni pseudoetiquetas, ni adaptar estadísticas, ni ajustar con su puntaje).
- Lateralidad: M02 es zurdo y M03/M04 diestros. Entrenar con espejo y promediar predicciones original+espejo (ver specs/design.md, decisión 5).
- Los `Clip` son únicos por archivo: no se pueden agrupar vistas; usar LOSO interno para elegir épocas.
- Transfer learning permitido citando el origen.
- Código reproducible (entrenamiento + inferencia), semillas fijas.
- Entrega: pesos, código, reporte técnico corto y CSV de predicciones.

## Clases
0 Café, 1 Aromática, 2 Hola, 3 Qué necesitas, 4 Con gusto.
Carpetas/etiquetas: sign_01 a sign_05 (mismo orden). Los .mp4 y .npy comparten nombre base.
Nombre en train: `S##_M##_V##_C##_Clip####.mp4` (seña, participante M01–M04, vista V01–V03, cámara, clip).
Participantes públicos: M02, M03, M04 (el test es M01).
Landmarks `.npy` ya vienen dados: forma esperada (n_fotogramas, 1662); la longitud varía. Inspeccionar antes de programar.
Entrega: `submission.csv` con columnas exactas `video_id,predicted_sign` (UTF-8, sin índice), una fila por ID de `submission_template.csv`.

## Convenciones
- Python 3.12, PyTorch. Dependencias en `requirements.txt`.
- Los datos (videos, landmarks) NO se commitean; van en `data/` (ignorado por git).
- Una rama por cambio; PR a main; sin commits directos.
- Fechas: presentación 19 nov 2026, clausura 20 nov.
