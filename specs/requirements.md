# Requisitos

Formato: WHEN/THE SYSTEM SHALL. Cada requisito es verificable.

## R1. Datos y split
- R1.1 THE SYSTEM SHALL parsear seña, participante (M01–M04), vista y clip desde el nombre de archivo, y verificar el emparejamiento video–landmarks y la forma de los .npy.
- R1.2 WHEN se arman folds, THE SYSTEM SHALL usar LOSO sobre los 3 participantes públicos (3 folds); ningún participante aparece en train y val a la vez.
- R1.2b WHERE se necesite un split interno (p. ej. elegir épocas), THE SYSTEM SHALL usar LOSO interno entre los participantes de entrenamiento: los números de Clip son únicos por archivo y no permiten agrupar las vistas de una misma ejecución, por lo que un split aleatorio filtraría vistas.
- R1.3 THE SYSTEM SHALL generar un manifiesto CSV (ruta, clase, firmante, vista, cámara, clip).
- R1.4 THE SYSTEM SHALL excluir `data/` del control de versiones.

## R2. Landmarks
- R2.1 THE SYSTEM SHALL usar los `.npy` oficiales (n_fotogramas, 1662: pose 132, rostro 1.404, manos 63+63) y manejar longitudes variables.
- R2.2 WHERE falten landmarks o haya valores faltantes, THE SYSTEM SHALL detectarlos y reportarlos antes de entrenar. (Re-extraer con MediaPipe Holistic solo si hace falta.)
- R2.3 THE SYSTEM SHALL normalizar (centrar en torso, escalar por ancho de hombros) y remuestrear a longitud fija T; toda estadística se estima solo con el train del fold.

## R3. Modelo sobre landmarks
- R3.1 THE SYSTEM SHALL entrenar un clasificador de secuencias (GRU/LSTM y Transformer pequeño) sobre landmarks.
- R3.2 THE SYSTEM SHALL reportar accuracy top-1 y F1 macro por fold de firmante.
- R3.3 WHERE se use aumento de datos (ruido, deformación temporal, espejo), THE SYSTEM SHALL poder desactivarlo por configuración.
- R3.4 THE SYSTEM SHALL ser robusto a la lateralidad: M02 es zurdo y M03/M04 diestros. Se entrena con cada muestra y su versión espejada (x -> 1-x, intercambio de manos y pares izquierda/derecha de pose y rostro) y, al predecir, se promedian las probabilidades de la muestra y su espejo. Es una transformación determinista por muestra, sin estadísticas del test.

## R4. Modelo sobre video
- R4.1 THE SYSTEM SHALL entrenar un modelo sobre frames con backbone preentrenado (transfer learning), citando su origen.
- R4.2 THE SYSTEM SHALL usar los mismos folds por firmante que R3 para comparar de forma justa.

## R5. Evaluación y entrega
- R5.1 THE SYSTEM SHALL calcular accuracy top-1 como métrica principal y F1 macro como secundaria.
- R5.2 WHEN se corre inferencia sobre test, THE SYSTEM SHALL generar `submission.csv` con columnas exactas `video_id,predicted_sign` (UTF-8, sin índice), una fila por ID de `submission_template.csv`, sin duplicados ni faltantes; `predicted_sign` en sign_01..sign_05.
- R5.3 THE SYSTEM SHALL permitir reproducir entrenamiento e inferencia con un solo comando documentado y semillas fijas.
- R5.4 THE SYSTEM SHALL usar el test solo para inferencia con un modelo ya fijado: sin entrenar, pseudoetiquetar, adaptar estadísticas ni ajustar con su puntaje.
- R5.5 THE SYSTEM SHALL conservar documentación de datos, preprocesamiento, validación, modelos y criterios de selección, por si los organizadores la piden.

## R6. Reporte
- R6.1 THE SYSTEM SHALL incluir un reporte técnico corto: datos, método, resultados por fold, limitaciones, fuentes de transfer learning.
- R6.2 THE SYSTEM SHALL reconocer en el reporte las limitaciones del manual: 5 señas, 4 personas, test de una sola persona, vistas correlacionadas.
