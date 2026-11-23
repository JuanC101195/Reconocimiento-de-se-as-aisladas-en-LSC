#!/usr/bin/env python3
"""
LSC5 - Inferencia final sobre el conjunto TEST.

Carga los 9 checkpoints previamente entrenados,
procesa los landmarks y genera submission.csv.
No realiza entrenamiento.
"""

from pathlib import Path
import torch
import train_lsc5 as lsc5


def main():

    # Rutas del proyecto
    home = Path.home()

    data_root = home / "Dataset"
    test_dir = data_root / "test" / "landmarks"

    out_dir = home / "resultados" / "lsc5_t10_ens3_cronos"

    template_path = data_root / "test" / "submission_template.csv"

    # Semillas utilizadas durante el entrenamiento
    seeds = [42, 7, 123]

    # Configuración del dispositivo
    torch.set_num_threads(4)
    device = torch.device("cpu")

    print("=" * 65)
    print("LSC5 - PREDICCIONES SOBRE TEST")
    print("=" * 65)

    print(f"Dataset: {data_root}")
    print(f"Test landmarks: {test_dir}")
    print(f"Checkpoints: {out_dir}")
    print(f"Template: {template_path}")

    # Comprobar rutas
    if not test_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio de test: {test_dir}")

    if not template_path.is_file():
        raise FileNotFoundError(f"No existe el submission template: {template_path}")

    if not out_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio de checkpoints: {out_dir}")

    # Comprobar los 9 checkpoints
    checkpoint_names = [
        f"{model}_final_seed{seed}.pt"
        for model in ["transformer", "cnn1d", "linear"]
        for seed in seeds
    ]

    missing = [name for name in checkpoint_names if not (out_dir / name).is_file()]

    if missing:
        raise FileNotFoundError(f"Faltan checkpoints: {missing}")

    print("\nLos 9 checkpoints están disponibles.")
    print("Comenzando inferencia...\n")

    # Generar predicciones usando los modelos entrenados
    lsc5.run_test_inference(
        test_dir=test_dir,
        seeds=seeds,
        out_dir=out_dir,
        data_root=data_root,
        template_path=template_path,
        device=device,
    )

    submission = out_dir / "submission.csv"

    if not submission.is_file():
        raise RuntimeError("No se generó submission.csv correctamente.")

    print("\n" + "=" * 65)
    print("INFERENCIA FINALIZADA")
    print(f"Archivo generado: {submission}")
    print("=" * 65)


if __name__ == "__main__":
    main()
