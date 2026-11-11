# Resultados (validación LOSO, sin tocar el test)

Participantes públicos: M02, M03, M04 (el test es M01). Métrica: accuracy top-1; macro-F1 como apoyo.
Todas las corridas fueron en CPU y con recursos reducidos; no son la corrida final.

## 1. Notebook original (RUTENCO), sin espejo
TCN multistream sobre landmarks, con LOSO interno para elegir épocas (30 épocas máx., paciencia 6, 1 semilla).

| Reservado | SVM | TCN |
|---|---|---|
| M02 | 0,208 | 0,424 |
| M03 | 0,869 | 0,901 |
| M04 | 0,819 | 0,824 |
| Media | 0,632 | 0,716 (±0,209) |

## 2. Diagnóstico de M02: es zurdo
Presencia de mano detectada (fracción de fotogramas), `scripts/diagnostico_lateralidad.py`:

| Participante | Mano izquierda | Mano derecha |
|---|---|---|
| M02 | 0,58 | 0,19 |
| M03 | 0,26 | 0,60 |
| M04 | 0,28 | 0,64 |

En sign_01 y sign_03, M02 usa solo la izquierda y M03/M04 solo la derecha. Un modelo entrenado con dos diestros nunca ve esas señas en espejo, de ahí el recall 0 en esas clases.

## 3. Experimentos de lateralidad (10 épocas fijas, media de 2 semillas)

| Variante | M02 | M03 | M04 | Media |
|---|---|---|---|---|
| Base (sin espejo) | 0,449 | 0,909 | 0,796 | 0,718 |
| Aumento por espejo | 0,917 | 0,929 | 0,847 | 0,898 |
| Aumento + promedio original/espejo (TTA) | 0,947 | 0,928 | 0,863 | 0,912 |
| Canonicalizar a mano dominante derecha | 0,908 | 0,925 | 0,828 | 0,887 |
| Canonicalizar + aumento | 0,939 | 0,919 | 0,897 | 0,918 |

Lo sólido es el salto de "base" a cualquier variante con espejo. El orden entre las variantes con espejo no es concluyente (2 semillas, diferencias de 1-3 puntos). Se adoptó aumento + TTA porque no depende de adivinar la mano dominante con un umbral.

## 4. Notebook final (`notebooks/LSC5_TCN_LOSO_espejo.ipynb`)
Espejo + TTA, LOSO interno, 12 épocas máx., paciencia 4, 1 semilla, CPU.

| Reservado | SVM (con espejo) | TCN (con espejo + TTA) |
|---|---|---|
| M02 | 0,936 | 0,949 |
| M03 | 0,888 | 0,920 |
| M04 | 0,904 | 0,880 |
| Media | 0,909 (±0,020) | 0,916 (±0,028) |

Macro-F1 OOF del TCN: 0,912. Épocas elegidas por fold: 8, 8 y 3.

## Lectura honesta
- El espejo explica casi toda la mejora: de ~0,72 a ~0,91 de media, y el peor participante pasa de ~0,45 a ~0,92.
- Con espejo, **el SVM y el TCN quedan casi empatados** (0,909 vs 0,916). La ventaja del TCN es pequeña y está dentro del ruido. Falta repetir con más semillas y en GPU antes de afirmar que el TCN es mejor.
- El rango entre participantes se redujo (0,88-0,95), pero M01 (test) sigue siendo una incógnita.

## Pendiente
- Corrida completa en Kaggle/clúster UNAB (más épocas, varias semillas, `TCN_CHANNELS` y `SEQ_LEN` mayores) con los tres folds.
- Modelo de video como complemento, si el problema residual está en los landmarks.
- Mirar los errores restantes (sign_05 se confunde con sign_03/sign_04).
- Inferencia sobre test solo con el modelo congelado (`RUN_FINAL_TEST_INFERENCE = True`).
