# Resultados (validación LOSO, sin tocar el test)

Participantes públicos: M02, M03, M04 (el test es M01). Métrica: accuracy top-1; macro-F1 como apoyo.
Las secciones 1 a 4 son corridas en CPU con recursos reducidos (no finales). La sección 5 (T7) y las secciones 6 y 7 (T9) son corridas en GPU (Kaggle, Tesla T4) con tres semillas.

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

## 5. Transformer compacto sobre landmarks (T7)
Notebook: `notebooks/LSC5_Transformer_LOSO_espejo.ipynb`. Mismo pipeline de datos, espejo + TTA y LOSO anidado que el TCN de la sección 4; solo cambia el modelo y los hiperparámetros. Kaggle, GPU Tesla T4, **3 semillas** (42, 7, 123).

`LSC5_Transformer` (301.286 parámetros): proyección por ramas anatómicas (manos 64, cuerpo 32, rostro 32 = `D_MODEL` 128), posición aprendible, `nn.TransformerEncoder` de 2 capas (4 cabezas, FFN 256, dropout 0,3, GELU, pre-LayerNorm), pooling por atención + promedio y clasificador lineal. `SEQ_LEN` 96, batch 32, máx. 80 épocas, paciencia 15, AdamW (lr 5e-4, weight decay 1e-2). Épocas finales: 31 (mediana de los LOSO internos). Una sola configuración: **no hubo barrido de hiperparámetros**.

Resumen (mismos folds LOSO, misma corrida de SVM con `SEQ_LEN` 96):

| Reservado | SVM (espejo + TTA) | TCN, sección 4 (CPU, 1 semilla) | Transformer (media de 3 semillas) | Transformer (ensamble de semillas) |
|---|---|---|---|---|
| M02 | 0,923 | 0,949 | 0,990 | 0,997 |
| M03 | 0,875 | 0,920 | 0,926 | 0,931 |
| M04 | 0,912 | 0,880 | 0,932 | 0,925 |
| Media | 0,903 (±0,021) | 0,916 (±0,028) | 0,950 (±0,029) | 0,951 (±0,033) |

Macro-F1 medio: SVM 0,903; Transformer 0,949 (media de semillas) y 0,950 (OOF del ensamble). La desviación entre semillas de la accuracy media LOSO es 0,004. Épocas elegidas por fold (M02, M03, M04): semilla 42 → 39, 39, 40; semilla 7 → 42, 12, 31; semilla 123 → 16, 30, 26.

## Lectura honesta
- El espejo explica casi toda la mejora de la sección 4: de ~0,72 a ~0,91 de media, y el peor participante pasa de ~0,45 a ~0,92.
- Con el mismo protocolo de GPU (3 semillas, `SEQ_LEN` 96, 436 características), el Transformer (0,956 con ensamble de semillas; 0,952 de media entre semillas) supera al TCN (0,911; 0,909) y al SVM (0,903) en la media y en el peor participante. La comparación con el TCN es por fin válida: +4,5 puntos con ensamble de semillas y +4,2 de media entre semillas. Ese margen sale de M02 (+4,5) y M04 (+9,1); en M03 empatan (0,928 los dos).
- El TCN en GPU con 3 semillas (0,909) no supera al TCN de CPU de la sección 4 (0,916), así que aquel resultado no estaba subestimando al TCN. Además es inestable entre semillas (±0,013 frente a ±0,004 del Transformer).
- Las 36 características nuevas no mostraron mejora demostrable (+0,2 puntos de media entre semillas, +0,5 con el ensamble de semillas; ambos dentro del ruido) y no corrigieron M03/V01.
- Ningún ensamble superó al Transformer solo con una estimación honesta: el óptimo de la rejilla (0,961) es optimista y la selección anidada da 0,949. Con 3 participantes, 0,5 puntos son unas 6 muestras.
- La accuracy media esconde un riesgo concentrado: casi todo el error del Transformer está en tres combinaciones firmante×vista de `sign_05`. Si M01 (test) tiene una vista problemática en alguna clase, el costo sería de varios puntos de golpe. El rango entre folds (0,928 a 0,995) es la mejor estimación disponible, y M01 sigue siendo una incógnita.
- Con solo 3 participantes, diferencias de menos de ~1 punto (≈11 muestras de 1125) están dentro del ruido.

## Pendiente
- **Decisión propuesta para T10:** congelar el **Transformer solo** (3 semillas, 436 características, 25 épocas finales). Los tres checkpoints `.pt` ya están entrenados; SVM y TCN no tienen modelo final porque su peso en el ensamble es 0. La inferencia sobre test debe cargar esos `.pt` (notebook corto de inferencia), sin reentrenar.
- Resolver o mitigar el fallo de `sign_05` en M03/V01 (y parcialmente M04/V02). Experimento más directo: quitar o perturbar las máscaras de presencia de manos y medir el efecto con el mismo LOSO y las mismas semillas. Como se diseña mirando los errores de esos folds, el resultado quedaría algo optimista y debe reportarse así.
- Modelo de video como complemento si el problema residual está en los landmarks (T5, T8).
- Reportar en el README la configuración final y dónde quedan los pesos (no se versionan).
