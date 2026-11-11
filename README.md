# Reconocimiento de señas aisladas en LSC

Proyecto del LatAm AI Challenge 2026 (UNAB, Chile – UdeA, Colombia): clasificar videos de señas aisladas de la Lengua de Señas Colombiana (dataset LSC5, 5 clases) con Deep Learning.

- Reto: https://www.latamiachallenge.com/dataset.php
- Especificaciones: [specs/requirements.md](specs/requirements.md), [specs/design.md](specs/design.md), [specs/tasks.md](specs/tasks.md)
- Resultados y diagnóstico: [reports/resultados.md](reports/resultados.md)
- Reglas y convenciones para contribuir: [CLAUDE.md](CLAUDE.md)

## Equipo
Marcos Contreras Cortés (UNAB), Cristobal Magnata Veragua (UNAB), Juan Esteban Cardozo Rivera (UdeA), Daniel Ramirez Cardenas (UdeA).

## Contenido
- `LSC5_Kaggle_TCN_LOSO.ipynb`: notebook original (TCN multistream sobre landmarks, LOSO).
- `notebooks/LSC5_TCN_LOSO_espejo.ipynb`: versión con espejo por lateralidad y LOSO interno. Es la versión recomendada.
- `scripts/diagnostico_lateralidad.py`: diagnóstico de mano dominante por participante.

## Hallazgo principal
M02 firma con la mano izquierda y M03/M04 con la derecha. Entrenar con versiones espejadas y promediar las predicciones de la muestra y su espejo subió la accuracy LOSO media de ~0,72 a ~0,91 (corrida reducida en CPU). Ver [reports/resultados.md](reports/resultados.md).

## Cómo ejecutar
Kaggle: agregar el dataset oficial con Add Input y ejecutar el notebook. Local: fijar `TRAIN_LANDMARKS_DIR` a la carpeta `train/landmarks`. Los datos nunca se suben al repo (`data/` está en `.gitignore`) ni se redistribuyen.

## Estado
Validación LOSO hecha en CPU con recursos reducidos. Pendiente: corrida completa en GPU/clúster UNAB, modelo de video y la inferencia final sobre test con el modelo congelado. Ver `specs/tasks.md`.
