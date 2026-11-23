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
- `notebooks/LSC5_TCN_LOSO_espejo.ipynb`: versión con espejo por lateralidad y LOSO interno.
- `notebooks/LSC5_T10_v2_Ensamble3_LOSO.ipynb`: ensamble de 3 modelos (`LSC5_Transformer`, `LSC5_CNN1D` y `LinearSegmentModel` con rotación 2D) con preprocesamiento `mask_aware`, recorte activo, espejo + TTA y LOSO anidado.
- `scripts/diagnostico_lateralidad.py`: diagnóstico de mano dominante por participante.
- `train_lsc5.py` y `job_ensamble.sbatch`: entrenamiento y trabajo por lotes en Slurm para el ensamble de 3 modelos.
- `predecir_test.py` y `job_predecir_test.sbatch`: inferencia sobre `test` cargando los 9 checkpoints `.pt` (`3` arquitecturas × `3` semillas `[42, 7, 123]`) para generar `submission.csv`.

## Hallazgo principal
M02 firma con la mano izquierda y M03/M04 con la derecha. Entrenar con versiones espejadas y promediar las predicciones de la muestra y su espejo subió la accuracy LOSO media de ~0,72 a ~0,91 (corrida reducida en CPU), y el ensamble de 3 modelos con LOSO anidado alcanzó 0,999 de accuracy OOF en GPU y 0,997 en CPU. Ver [reports/resultados.md](reports/resultados.md).

## Cómo ejecutar
- **Kaggle / local:** agregar o apuntar al dataset oficial (`train/landmarks` y `test/landmarks`) y ejecutar el notebook o script correspondiente. Los datos (`data/`) y los checkpoints pesados (`.pt`, `.npy`) no se suben al repositorio.
- **Clúster `cronos` (Slurm):** activar el entorno (`~/venv_entrenamiento/bin/activate`), ubicar los scripts en `~/lsc5_ensamble` y enviar los trabajos con `sbatch`:
  - Entrenamiento / checkpoints del ensamble: `sbatch job_ensamble.sbatch` (genera los 9 `.pt` en `~/resultados/lsc5_t10_ens3_cronos`).
  - Inferencia sobre `test`: `sbatch job_predecir_test.sbatch` (genera `~/resultados/lsc5_t10_ens3_cronos/submission.csv` con 375 filas).

## Estado
Validación LOSO anidada completada y corrida final en el clúster `cronos` ejecutada con el modelo congelado, generando `submission.csv` sobre `test` a partir de los 9 checkpoints entrenados. Los pesos `.pt` se respaldan fuera del control de versiones según lo indicado para la entrega. Ver `specs/tasks.md` y [reports/resultados.md](reports/resultados.md).