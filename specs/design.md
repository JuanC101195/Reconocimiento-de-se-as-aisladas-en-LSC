# Diseño

## Estructura propuesta
```
data/            # NO versionado: videos, landmarks .npy, manifiesto
src/
  data/          # parseo de nombres, manifiesto, folds por firmante
  features/      # extracción MediaPipe, normalización, remuestreo
  models/        # landmarks (GRU/Transformer) y video (backbone preentrenado)
  train.py       # entrenamiento con config + semilla
  predict.py     # inferencia -> CSV
configs/         # yaml por experimento
notebooks/       # exploración y notebook de entrega (.ipynb)
reports/         # resultados por fold, reporte técnico
specs/
```

## Decisiones
1. **Validación LOSO.** Solo hay 3 participantes públicos (M01–M04 menos el de test), así que son 3 folds. Cada fold se reporta por separado, no solo el promedio.
2. **Las 3 vistas.** Las vistas son grabaciones simultáneas de la misma ejecución: están correlacionadas y no son muestras independientes. Siempre juntas en cualquier split.
3. **Landmarks primero.** Es barato de entrenar, rápido de iterar y cabe en Colab/Kaggle durante la 1ª semana. El modelo de video va en paralelo y se entrena a fondo en el clúster UNAB.
4. **Comparación justa.** Ambos enfoques usan los mismos folds y la misma métrica; un ensamble final se decide por resultados.
5. **Lateralidad (hallazgo clave).** M02 firma con la mano izquierda (presente en ~58 % de los fotogramas vs ~19 % la derecha) y M03/M04 con la derecha (~60-64 % vs ~26-28 %). Sin espejo, el LOSO con M02 reservado daba ~0,45 de accuracy (sign_01 y sign_03 con recall 0). Con aumento por espejo y promedio de predicciones sube a ~0,95 y la media LOSO de ~0,72 a ~0,91. El firmante de test (M01) puede ser zurdo o diestro, por eso se trata de forma simétrica. Detalle en `reports/resultados.md`.
6. **Split interno.** Los `Clip` son únicos por archivo (no hay forma de agrupar vistas de una ejecución con los metadatos), así que la elección de épocas usa LOSO interno.
7. **Reproducibilidad.** Config yaml por experimento, semillas fijas, resultados guardados en `reports/`.

## Riesgos
- Solo 3 participantes de entrenamiento: riesgo alto de sobreajuste al firmante. Mitigar con aumento de datos (espejo) y LOSO.
- El resultado depende de cuánto se parezca M01 a los demás; el rango entre folds es la mejor estimación de ese riesgo.
- Uso y distribución de los datos: solo fines académicos, sin redistribuir; el repo nunca contiene datos.
- Criterios oficiales de puntuación y desempate aún por anunciar (macro-F1 y accuracy son los candidatos).
- Fuga por el test: ni siquiera estadísticas de normalización pueden calcularse con él.
