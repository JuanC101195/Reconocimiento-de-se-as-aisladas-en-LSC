# Resultados (validación LOSO, sin tocar el test)

Participantes públicos: M02, M03, M04 (el test es M01). Métrica: accuracy top-1; macro-F1 como apoyo.
Las secciones 1 a 4 son corridas en CPU con recursos reducidos (no finales). La sección 5 (T7) y las secciones 6 y 7 (T9) son corridas en GPU (Kaggle, Tesla T4) con tres semillas. La sección 8 (T10 v2) corresponde a corridas en Kaggle con GPU Tesla T4 y con CPU (4 threads) usando tres semillas (`[42, 7, 123]`).

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

## 6. Comparación de modelos sobre landmarks (T9)
Notebook: `notebooks/LSC5_Transformer_TCN_SVM_LOSO_espejo.ipynb`. Kaggle, GPU Tesla T4, 3 semillas (42, 7, 123), `SEQ_LEN` 96, 436 características, espejo + TTA, LOSO anidado.

| Modelo | Accuracy OOF (ensamble de semillas) |
|---|---|
| SVM | 0,903 |
| TCN | 0,911 |
| Transformer | 0,956 |
| Promedio uniforme | 0,954 |

Transformer por participante reservado: M02 0,995; M03 0,928; M04 0,947.

## 7. Selección del ensamble
Regla: se usa un ensamble solo si su estimación anidada supera al mejor modelo individual por al menos 0,005.

- Mejor combinación de la rejilla: 0,961.
- Selección anidada: 0,949 (M02 1,000; M03 0,915; M04 0,933).
- Mejor modelo individual (Transformer): 0,956.

Resultado de la regla: modelo individual (Transformer; pesos SVM 0, TCN 0, Transformer 1). Épocas finales: 25. Checkpoints `.pt`: 3 (uno por semilla).

Errores del modelo final: 49 de 1125, de los cuales 41 son de `sign_05` (recall 0,82), confundida con `sign_04` (27) y `sign_02` (14).

## 8. Ensamble de 3 modelos con preprocesamiento `mask_aware` 2D sin rostro (T10 v2)
Notebook: `LSC5_T10_v2_Ensamble3_LOSO.ipynb`. Mantiene el preprocesamiento `mask_aware` de T10 (máscaras crudas de presencia, interpolación de huecos cortos hasta 3 frames, `SEQ_LEN` 32, espejo + TTA y LOSO anidado), sumando recorte activo de movimiento (`trim_active_motion_pose`) y anulación de coordenadas `Z` y faciales (`ZERO_COLS`). Ya no se usa un flag manual `keep_hands` porque la presencia de manos queda fijada por las máscaras crudas (`MASK_FEATURES`) y la rama de manos del Transformer.

Combina en promedio uniforme (`1/3 + 1/3 + 1/3`) tres arquitecturas compactas (137.512 parámetros en total): `LSC5_Transformer` (84.034 parámetros; ramas 40/20/4, `D_MODEL` 64, 2 capas, 4 cabezas, FFN 128), `LSC5_CNN1D` (38.213 parámetros; convoluciones 1D residuales *depthwise-separable* con kernels 5 y 3) y `LinearSegmentModel` (15.265 parámetros; 4 tramos temporales + std global, entrenado con rotación 2D aleatoria de ±10° y 50 épocas). Ejecutado en Kaggle con **3 semillas** (`42, 7, 123`) tanto en **GPU Tesla T4** (`24,46 min`, `AMP: True`) como en **CPU, 4 threads** (`55,47 min`, `AMP: False`).

| Reservado / Métrica | Transformer (`84.034` p., GPU) | CNN1D (`38.213` p., GPU) | Lineal 4 tramos + Rot2D (`15.265` p., GPU) | Ensamble 3 modelos (GPU Tesla T4) | Ensamble 3 modelos (CPU, 4 threads) |
|---|---|---|---|---|---|
| M02 (Accuracy) | 0,984 | 0,995 | 1,000 | **1,000** | **1,000** |
| M03 (Accuracy) | 1,000 | 0,936 | 0,984 | **1,000** | **1,000** |
| M04 (Accuracy) | 0,912 | 0,997 | 0,957 | **0,997** | **0,992** |
| Accuracy OOF (1.125) | 0,965 | 0,976 | 0,980 | **0,999** (1 error) | **0,997** (3 errores) |
| Macro-F1 OOF | 0,966 | 0,976 | 0,980 | **0,999** | **0,997** |

Épocas finales medianas (LOSO interno, GPU T4): Transformer 24, CNN1D 17, Lineal 50 (1 error en `M04/V03/sign_04`, recall 0,96). En CPU (4 threads): épocas finales medianas Transformer 20, CNN1D 20, Lineal 50 (3 errores en `M04/V03/sign_04`, recall 0,88).

## Lectura honesta
- El espejo explica casi toda la mejora de la sección 4: de ~0,72 a ~0,91 de media, y el peor participante pasa de ~0,45 a ~0,92.
- Con el mismo protocolo de GPU (3 semillas, `SEQ_LEN` 96, 436 características), el Transformer (0,956 con ensamble de semillas; 0,952 de media entre semillas) supera al TCN (0,911; 0,909) y al SVM (0,903) en la media y en el peor participante. La comparación con el TCN es por fin válida: +4,5 puntos con ensamble de semillas y +4,2 de media entre semillas. Ese margen sale de M02 (+4,5) y M04 (+9,1); en M03 empatan (0,928 los dos).
- El TCN en GPU con 3 semillas (0,909) no supera al TCN de CPU de la sección 4 (0,916), así que aquel resultado no estaba subestimando al TCN. Además es inestable entre semillas (±0,013 frente a ±0,004 del Transformer).
- Las 36 características nuevas no mostraron mejora demostrable (+0,2 puntos de media entre semillas, +0,5 con el ensamble de semillas; ambos dentro del ruido) y no corrigieron M03/V01.
- Ningún ensamble superó al Transformer solo con una estimación honesta: el óptimo de la rejilla (0,961) es optimista y la selección anidada da 0,949. Con 3 participantes, 0,5 puntos son unas 6 muestras.
- En T10 v2 (3 semillas `[42, 7, 123]`), combinar en partes iguales `Transformer` (`84.034` parámetros), `CNN1D` (`38.213` parámetros) y `Lineal 4 tramos + Rot2D` (`15.265` parámetros) con recorte activo y proyección 2D sin rostro baja los errores OOF a 1/1.125 en GPU (`0,9991`) y 3/1.125 en CPU (`0,9973`), compensando las fallas puntuales de cada modelo por separado.
- El conteo total bajó de 301.286 parámetros del Transformer de T7/T9 a **137.512 parámetros** en el ensamble actual (`84.034` para Transformer + `38.213` para CNN1D + `15.265` para Lineal), reduciendo la capacidad por rama del Transformer (`D_MODEL` de 128 a 64) para evitar sobreajuste y repartiendo el cómputo en arquitecturas ligeras; en CPU de 4 threads el tiempo sube a `55,47 min` frente a `24,46 min` en GPU T4 por ausencia de AMP y menor paralelismo.
- La accuracy media esconde un riesgo concentrado: casi todo el error del Transformer está en tres combinaciones firmante×vista de `sign_05`. Si M01 (test) tiene una vista problemática en alguna clase, el costo sería de varios puntos de golpe. El rango entre folds (0,928 a 0,995) es la mejor estimación disponible, y M01 sigue siendo una incógnita.
- Con solo 3 participantes, diferencias de menos de ~1 punto (≈11 muestras de 1125) están dentro del ruido.

## Pendiente
- Conservar y respaldar localmente el paquete de checkpoints finales `.pt` (3 modelos × 3 semillas `[42, 7, 123]`) y artefactos OOF generados con `TRAIN_FINAL_MODEL=True` en GPU T4 (`lsc5_t10_ens3_a8c94d8c1fad_artifacts.zip`), evitando depender de sesiones en CPU si no es necesario.
- Evaluar la inferencia sobre test (`RUN_FINAL_TEST_INFERENCE=True`) únicamente cuando se decida congelar y enviar la predicción oficial sobre `M01`.
- Modelo de video como complemento si el problema residual está en los landmarks (T5, T8).
- Reportar en el README la configuración final y dónde quedan los pesos (no se versionan).
