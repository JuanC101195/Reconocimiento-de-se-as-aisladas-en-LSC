#!/usr/bin/env python3
"""
LSC5 | T10 v2: Ensamble de 3 arquitecturas (Transformer compacto + Lineal 4 tramos con rotación 2D + CNN1D compacta)
Preprocesamiento mask_aware + recorte activo + 2D sin rostro + espejo/TTA + LOSO anidado.

Soporta ejecución en:
  - PC local con GPU (ej. RTX 3060) o CPU: --device auto|cuda|cpu
  - Clúster cronos (CPU + Slurm + CodeCarbon): --device cpu --threads 8 --use-codecarbon
"""

import argparse
import json
import math
import random
import re
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


# ========= Constantes de representación y arquitectura =========
CLASS_NAMES = [f"sign_{i:02d}" for i in range(1, 6)]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: c for i, c in enumerate(CLASS_NAMES)}
PUBLIC_SIGNERS_EXPECTED = ["M02", "M03", "M04"]

POSE_SLICE = slice(0, 132)
FACE_SLICE = slice(132, 1536)
LEFT_HAND_SLICE = slice(1536, 1599)
RIGHT_HAND_SLICE = slice(1599, 1662)

BODY_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24]
FACE_JOINTS = [1, 33, 133, 362, 263, 61, 291, 13, 14, 70, 300]
BASE_FEATURE_DIM = 200
FEATURE_DIM = BASE_FEATURE_DIM * 2  # 400
EXTRA_DIM = 36
INPUT_DIM = FEATURE_DIM + EXTRA_DIM  # 436
MASK_FEATURES = [197, 198, 199, 397, 398, 399]

PAIRS = [
    (1, 4),
    (2, 5),
    (3, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20),
    (21, 22),
    (23, 24),
    (25, 26),
    (27, 28),
    (29, 30),
    (31, 32),
]
FACE_PAIRS = [(33, 263), (133, 362), (61, 291), (70, 300)]
FINGER_CHAINS = [
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
]

Z_IDX = list(range(2, 132, 3)) + list(range(134, 164, 4))
FACE_IDX = list(range(164, 197))
ZERO_COLS = Z_IDX + FACE_IDX + [i + 200 for i in Z_IDX + FACE_IDX]

XY_X_BASE = list(range(0, 132, 3)) + list(range(132, 164, 4))
XY_X_IDX = XY_X_BASE + [i + 200 for i in XY_X_BASE]
XY_Y_IDX = [i + 1 for i in XY_X_IDX]

HAND_STREAM_IDX = (
    list(range(0, 132)) + list(range(200, 332)) + [197, 198] + list(range(400, 436))
)
BODY_STREAM_IDX = list(range(132, 164)) + list(range(332, 364))
FACE_STREAM_IDX = list(range(164, 197)) + list(range(364, 397)) + [199]

PREPROCESS_VERSION = "t10_trim2d_ens3_v2"
SEQ_LEN = 32
MAX_INTERPOLATION_GAP = 3
N_SEGMENTS = 4

# Hiperparámetros
D_MODEL, N_LAYERS, N_HEADS, FFN_DIM = 64, 2, 4, 128
HAND_DIM, BODY_DIM, FACE_DIM = 40, 20, 4
DROPOUT = 0.3
LR = 5e-4
WEIGHT_DECAY = 1e-2
LIN_EPOCHS = 50
LIN_LR = 1e-3
LIN_WD = 1e-2
LIN_DROPOUT = 0.3


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========= Descubrimiento de datos y metadatos =========
def _class_dirs(path: Path):
    if not path.exists() or not path.is_dir():
        return []
    return [
        p for p in path.iterdir() if p.is_dir() and re.fullmatch(r"sign_\d{2}", p.name)
    ]


def discover_train_landmarks(root: Path, override=None) -> Path:
    if override is not None:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"La ruta --train-dir no existe: {p}")
        return p
    if not root.exists():
        raise FileNotFoundError(f"No existe --data-root: {root}")
    candidates = []
    for p in [root] + list(root.rglob("landmarks")):
        if not p.is_dir():
            continue
        if "test" in [part.lower() for part in p.parts]:
            continue
        if len(_class_dirs(p)) >= 5 and any(p.rglob("*.npy")):
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(
            f"No encontré train/landmarks con sign_01...sign_05 dentro de {root}. "
            "Usa --train-dir para indicar la carpeta exacta."
        )
    candidates.sort(
        key=lambda p: ("train" not in [x.lower() for x in p.parts], len(str(p)))
    )
    return candidates[0]


def discover_test_landmarks(root: Path, override=None) -> Path:
    if override is not None:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"La ruta --test-dir no existe: {p}")
        return p
    folders = [
        p
        for p in root.rglob("landmarks")
        if p.is_dir()
        and "test" in [part.lower() for part in p.parts]
        and any(p.rglob("*.npy"))
    ]
    if not folders:
        raise FileNotFoundError(
            f"No encontré test/landmarks en {root}; usa --test-dir."
        )
    return sorted(folders, key=lambda p: len(str(p)))[0]


def parse_sample_metadata(path: Path) -> dict:
    stem = path.stem
    parent_match = re.fullmatch(r"(sign_\d{2})", path.parent.name)
    sign_match = re.search(r"(?:^|_)S(\d{2})(?:_|$)", stem)
    label = (
        parent_match.group(1)
        if parent_match
        else (f"sign_{int(sign_match.group(1)):02d}" if sign_match else None)
    )
    signer_match = re.search(r"(?:^|_)M(\d{2})(?:_|$)", stem)
    if signer_match is None:
        raise ValueError(f"No pude extraer participante M## del nombre: {path.name}")
    signer = f"M{signer_match.group(1)}"
    view_match = re.search(r"(?:^|_)V(\d{2})(?:_|$)", stem)
    camera_match = re.search(r"(?:^|_)C(\d{2})(?:_|$)", stem)
    clip_match = re.search(r"Clip(\d+)", stem, flags=re.IGNORECASE)
    if label not in CLASS_NAMES:
        raise ValueError(f"Clase inválida en {path}")
    group_id = f"{label}_{signer}_Clip{clip_match.group(1)}" if clip_match else stem
    return {
        "path": str(path),
        "video_id": stem,
        "label": label,
        "y": CLASS_TO_ID[label],
        "signer": signer,
        "view": f"V{view_match.group(1)}" if view_match else "unknown",
        "camera": f"C{camera_match.group(1)}" if camera_match else "unknown",
        "clip": clip_match.group(1) if clip_match else None,
        "group_id": group_id,
    }


# ========= Preprocesamiento mask_aware + recorte activo + 2D sin rostro =========
def load_landmark_array(path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{path}: esperaba matriz 2D, recibí {arr.shape}")
    if arr.shape[1] != 1662 and arr.shape[0] == 1662:
        arr = arr.T
    if arr.shape[1] != 1662:
        raise ValueError(
            f"{path}: esperaba 1662 features por frame, recibí {arr.shape}"
        )
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if len(arr) < 2:
        raise ValueError(f"{path}: secuencia muy corta ({len(arr)} frames)")
    return arr


def _flipx(block: np.ndarray) -> np.ndarray:
    block = block.copy()
    present = np.abs(block).max(axis=2) > 1e-6
    block[..., 0] = np.where(present, 1.0 - block[..., 0], 0.0)
    return block


def mirror_raw(arr: np.ndarray) -> np.ndarray:
    t = len(arr)
    pose = _flipx(arr[:, POSE_SLICE].reshape(t, 33, 4))
    face = _flipx(arr[:, FACE_SLICE].reshape(t, 468, 3))
    lh = _flipx(arr[:, LEFT_HAND_SLICE].reshape(t, 21, 3))
    rh = _flipx(arr[:, RIGHT_HAND_SLICE].reshape(t, 21, 3))
    for i, j in PAIRS:
        pose[:, [i, j]] = pose[:, [j, i]]
    for i, j in FACE_PAIRS:
        face[:, [i, j]] = face[:, [j, i]]
    return np.concatenate(
        [
            pose.reshape(t, -1),
            face.reshape(t, -1),
            rh.reshape(t, -1),
            lh.reshape(t, -1),
        ],
        axis=1,
    ).astype(np.float32)


def raw_presence_masks(raw: np.ndarray) -> np.ndarray:
    t = len(raw)
    pose = raw[:, POSE_SLICE].reshape(t, 33, 4)
    face = raw[:, FACE_SLICE].reshape(t, 468, 3)
    lh = raw[:, LEFT_HAND_SLICE].reshape(t, 21, 3)
    rh = raw[:, RIGHT_HAND_SLICE].reshape(t, 21, 3)
    return np.stack(
        [
            np.max(np.abs(pose[:, :, :3]), axis=(1, 2)) > 1e-6,
            np.max(np.abs(face), axis=(1, 2)) > 1e-6,
            np.max(np.abs(lh), axis=(1, 2)) > 1e-6,
            np.max(np.abs(rh), axis=(1, 2)) > 1e-6,
        ],
        axis=1,
    )


def fill_short_missing_runs(
    block: np.ndarray, observed: np.ndarray, max_gap=MAX_INTERPOLATION_GAP
):
    block = np.asarray(block, dtype=np.float32)
    observed = np.asarray(observed, dtype=bool)
    out = block.copy()
    usable = observed.copy()
    valid_indices = np.flatnonzero(observed)
    for a, b in zip(valid_indices[:-1], valid_indices[1:]):
        count = int(b - a - 1)
        if 0 < count <= max_gap:
            for i in range(a + 1, b):
                w = (i - a) / (b - a)
                out[i] = (1.0 - w) * out[a] + w * out[b]
                usable[i] = True
    return out, usable


def resample_masked_block(
    block: np.ndarray, usable: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    lo = np.floor(positions).astype(np.int64)
    hi = np.minimum(lo + 1, len(block) - 1)
    alpha = (positions - lo).astype(np.float32)
    shape = (len(positions),) + (1,) * (block.ndim - 1)
    w = alpha.reshape(shape)
    mix = (1.0 - w) * block[lo] + w * block[hi]
    both = usable[lo] & usable[hi]
    choose_lo = usable[lo] & (~usable[hi]) & (alpha < 0.5)
    choose_hi = usable[hi] & (~usable[lo]) & (alpha >= 0.5)
    out = np.zeros_like(mix)
    out[both] = mix[both]
    out[choose_lo] = block[lo[choose_lo]]
    out[choose_hi] = block[hi[choose_hi]]
    return out.astype(np.float32)


def resample_landmarks_mask_aware(raw: np.ndarray, target_len=SEQ_LEN):
    raw = np.asarray(raw, dtype=np.float32)
    observed = raw_presence_masks(raw)
    positions = np.linspace(0, len(raw) - 1, target_len, dtype=np.float32)
    nearest = np.clip(np.floor(positions + 0.5).astype(np.int64), 0, len(raw) - 1)

    specs = [
        (POSE_SLICE, 0),
        (FACE_SLICE, 1),
        (LEFT_HAND_SLICE, 2),
        (RIGHT_HAND_SLICE, 3),
    ]
    result = np.zeros((target_len, 1662), dtype=np.float32)
    for sl, j in specs:
        repaired, usable = fill_short_missing_runs(raw[:, sl], observed[:, j])
        result[:, sl] = resample_masked_block(repaired, usable, positions)

    original_masks = observed[nearest][:, [2, 3, 1]].astype(np.float32)
    return result, original_masks


def trim_active_motion_pose(
    raw: np.ndarray, rest_rise=0.3, min_active=3, margin=2
) -> np.ndarray:
    t = len(raw)
    if t <= 10:
        return raw
    pose = raw[:, POSE_SLICE].reshape(t, 33, 4)
    ls, rs = pose[:, 11, :2], pose[:, 12, :2]
    both = (np.abs(ls).sum(axis=1) > 1e-6) & (np.abs(rs).sum(axis=1) > 1e-6)
    if not both.any():
        return raw
    width = max(float(np.linalg.norm(ls[both] - rs[both], axis=-1).mean()), 1e-3)

    wrists = pose[:, [15, 16], :2]
    w_ok = np.abs(wrists).sum(axis=2) > 1e-6
    if not w_ok.any():
        return raw

    wrists_y = np.where(w_ok, wrists[:, :, 1] / width, np.nan)
    rise = np.nanpercentile(wrists_y, 90, axis=0) - wrists_y
    active = (np.nan_to_num(rise, nan=0.0) > rest_rise).any(axis=1)

    sustained = (
        np.convolve(
            active.astype(np.int32), np.ones(min_active, dtype=np.int32), mode="valid"
        )
        == min_active
    )
    starts = np.flatnonzero(sustained)
    if len(starts) == 0:
        return raw
    first = max(int(starts[0]) - margin, 0)
    last = min(int(starts[-1]) + min_active + margin, t)
    return raw[first:last] if (last - first) >= 6 else raw


def landmarks_to_features(
    arr: np.ndarray, target_len=SEQ_LEN, presence_override=None
) -> np.ndarray:
    t = len(arr)
    assert t == target_len
    pose = arr[:, POSE_SLICE].reshape(t, 33, 4)
    face = arr[:, FACE_SLICE].reshape(t, 468, 3)
    left_hand = arr[:, LEFT_HAND_SLICE].reshape(t, 21, 3)
    right_hand = arr[:, RIGHT_HAND_SLICE].reshape(t, 21, 3)

    left_shoulder = pose[:, 11, :3]
    right_shoulder = pose[:, 12, :3]
    torso_center = (left_shoulder + right_shoulder) / 2.0
    shoulder_width = np.linalg.norm(
        left_shoulder[:, :2] - right_shoulder[:, :2], axis=1
    )
    valid_widths = shoulder_width[shoulder_width > 1e-4]
    body_scale = max(float(np.median(valid_widths)) if len(valid_widths) else 1.0, 1e-3)

    def hand_features(hand, observed):
        geometric_present = (np.max(np.abs(hand), axis=(1, 2)) > 1e-6).astype(
            np.float32
        )
        present = geometric_present if observed is None else observed
        wrist = hand[:, 0, :]
        local = (hand - wrist[:, None, :]) / body_scale
        global_wrist = (wrist - torso_center) / body_scale
        local[geometric_present == 0] = 0.0
        global_wrist[geometric_present == 0] = 0.0
        return local.reshape(t, -1), global_wrist, present

    left_local, left_wrist, left_present = hand_features(
        left_hand, None if presence_override is None else presence_override[:, 0]
    )
    right_local, right_wrist, right_present = hand_features(
        right_hand, None if presence_override is None else presence_override[:, 1]
    )

    body_xyz = (pose[:, BODY_JOINTS, :3] - torso_center[:, None, :]) / body_scale
    body_visibility = pose[:, BODY_JOINTS, 3:4]
    body_features = np.concatenate([body_xyz, body_visibility], axis=2).reshape(t, -1)

    selected_face = face[:, FACE_JOINTS, :]
    nose = face[:, 1, :]
    eye_width = np.linalg.norm(face[:, 33, :2] - face[:, 263, :2], axis=1)
    valid_eye_widths = eye_width[eye_width > 1e-4]
    face_scale = max(
        float(np.median(valid_eye_widths)) if len(valid_eye_widths) else 1.0, 1e-3
    )
    face_present = (np.max(np.abs(face), axis=(1, 2)) > 1e-6).astype(np.float32)
    face_local = (selected_face - nose[:, None, :]) / face_scale
    face_local[face_present == 0] = 0.0
    if presence_override is not None:
        face_present = presence_override[:, 2]
    face_features = face_local.reshape(t, -1)

    base = np.concatenate(
        [
            left_local,
            right_local,
            left_wrist,
            right_wrist,
            body_features,
            face_features,
            left_present[:, None],
            right_present[:, None],
            face_present[:, None],
        ],
        axis=1,
    ).astype(np.float32)

    velocity = np.diff(base, axis=0, prepend=base[:1]).astype(np.float32)
    velocity[:, -3:] = 0.0
    features = np.concatenate([base, velocity], axis=1).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _joint_angles(hand: np.ndarray) -> np.ndarray:
    out = []
    for chain in FINGER_CHAINS:
        for k in range(1, 4):
            p_prev, p_joint, p_next = (
                hand[:, chain[k - 1]],
                hand[:, chain[k]],
                hand[:, chain[k + 1]],
            )
            v1, v2 = p_prev - p_joint, p_next - p_joint
            denom = np.maximum(
                np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1), 1e-6
            )
            cos = np.clip(np.sum(v1 * v2, axis=1) / denom, -1.0, 1.0)
            out.append(np.arccos(cos) / np.pi)
    return np.stack(out, axis=1)


def extra_features(arr: np.ndarray) -> np.ndarray:
    t = len(arr)
    pose = arr[:, POSE_SLICE].reshape(t, 33, 4)
    face = arr[:, FACE_SLICE].reshape(t, 468, 3)
    hands = [
        arr[:, LEFT_HAND_SLICE].reshape(t, 21, 3),
        arr[:, RIGHT_HAND_SLICE].reshape(t, 21, 3),
    ]

    widths = np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=1)
    valid = widths[widths > 1e-4]
    scale = max(float(np.median(valid)) if len(valid) else 1.0, 1e-3)
    chest = pose[:, [11, 12, 23, 24], :2].mean(axis=1)
    face_present = np.max(np.abs(face), axis=(1, 2)) > 1e-6
    nose = face[:, 1, :2]
    mouth = (face[:, 13, :2] + face[:, 14, :2]) / 2.0

    blocks_ang, blocks_dist = [], []
    for hand in hands:
        present = np.max(np.abs(hand), axis=(1, 2)) > 1e-6
        ang = _joint_angles(hand)
        wrist = hand[:, 0, :2]
        dist = (
            np.stack(
                [
                    np.linalg.norm(wrist - nose, axis=1),
                    np.linalg.norm(wrist - mouth, axis=1),
                    np.linalg.norm(wrist - chest, axis=1),
                ],
                axis=1,
            )
            / scale
        )
        dist[~face_present, :2] = 0.0
        ang[~present] = 0.0
        dist[~present] = 0.0
        blocks_ang.append(ang)
        blocks_dist.append(dist)
    out = np.concatenate(
        [blocks_ang[0], blocks_ang[1], blocks_dist[0], blocks_dist[1]], axis=1
    )
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def full_features(raw: np.ndarray, seq_len=SEQ_LEN) -> np.ndarray:
    raw = trim_active_motion_pose(raw, rest_rise=0.3, min_active=3, margin=2)
    arr, original_masks = resample_landmarks_mask_aware(raw, seq_len)
    base = landmarks_to_features(arr, seq_len, presence_override=original_masks)
    out = np.concatenate([base, extra_features(arr)], axis=1)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    out[:, ZERO_COLS] = 0.0
    assert out.shape == (seq_len, INPUT_DIM) and np.isfinite(out).all()
    return out


def build_feature_tensor(meta_df: pd.DataFrame, mirror=False) -> np.ndarray:
    sequences = []
    desc = "Espejando landmarks" if mirror else "Procesando landmarks"
    for row in tqdm(meta_df.itertuples(index=False), total=len(meta_df), desc=desc):
        raw = load_landmark_array(row.path)
        if mirror:
            raw = mirror_raw(raw)
        sequences.append(full_features(raw, SEQ_LEN))
    return np.stack(sequences).astype(np.float32)


def seq_to_segment_vector(X_seq: np.ndarray, n_segments=N_SEGMENTS) -> np.ndarray:
    segs = np.array_split(X_seq, n_segments, axis=1)
    means = [s.mean(axis=1) for s in segs]
    std = X_seq.std(axis=1)
    return np.concatenate(means + [std], axis=1).astype(np.float32)


def batch_rotate_2d_torch(xb: torch.Tensor, max_deg=10.0, p=0.70) -> torch.Tensor:
    """Aplica rotación 2D aleatoria (±max_deg) por muestra en un batch (B, T, D)."""
    B = xb.shape[0]
    device = xb.device
    apply_mask = torch.rand(B, device=device) < p
    if not apply_mask.any():
        return xb
    xb = xb.clone()
    angles = (torch.rand(B, device=device) * 2.0 - 1.0) * math.radians(max_deg)
    angles = torch.where(apply_mask, angles, torch.zeros_like(angles))
    c = torch.cos(angles).view(B, 1, 1)
    s = torch.sin(angles).view(B, 1, 1)
    vx = xb[:, :, XY_X_IDX]
    vy = xb[:, :, XY_Y_IDX]
    xb[:, :, XY_X_IDX] = c * vx - s * vy
    xb[:, :, XY_Y_IDX] = s * vx + c * vy
    return xb


def batch_seq_to_segment_vector_torch(
    xb: torch.Tensor, n_segments=N_SEGMENTS
) -> torch.Tensor:
    chunks = torch.tensor_split(xb, n_segments, dim=1)
    means = [c.mean(dim=1) for c in chunks]
    std = xb.std(dim=1, unbiased=False)
    return torch.cat(means + [std], dim=1)


# ========= Escalado, métricas y auditoría de errores =========
def fit_sequence_scaler(X_train: np.ndarray) -> dict:
    flat = X_train.reshape(-1, X_train.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std[std < 1e-5] = 1.0
    return {"mean": mean, "std": std}


def apply_sequence_scaler(X_data: np.ndarray, scaler: dict) -> np.ndarray:
    return (
        (X_data - scaler["mean"][None, None, :]) / scaler["std"][None, None, :]
    ).astype(np.float32)


def calculate_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def print_error_audit(
    meta_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, title: str
):
    df = meta_df[["signer", "view", "label"]].copy()
    df["err"] = (y_true != y_pred).astype(int)
    agg = df.groupby(["signer", "view", "label"], as_index=False).agg(
        n=("err", "size"),
        errors=("err", "sum"),
    )
    agg["recall"] = (agg["n"] - agg["errors"]) / agg["n"]
    bad = agg[agg["errors"] > 0].sort_values(
        ["errors", "signer", "view"], ascending=[False, True, True]
    )
    print(f"\n--- Auditoría de errores: {title} ---")
    if len(bad) == 0:
        print("  ¡0 errores en todas las combinaciones!")
    else:
        print(bad.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# ========= Arquitecturas =========
def _branch(n_in: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, n_out), nn.LayerNorm(n_out), nn.GELU())


class LSC5_Transformer(nn.Module):
    def __init__(self, dropout=DROPOUT, num_classes=5):
        super().__init__()
        self.register_buffer(
            "hand_idx", torch.tensor(HAND_STREAM_IDX, dtype=torch.long)
        )
        self.register_buffer(
            "body_idx", torch.tensor(BODY_STREAM_IDX, dtype=torch.long)
        )
        self.register_buffer(
            "face_idx", torch.tensor(FACE_STREAM_IDX, dtype=torch.long)
        )
        self.hand_proj = _branch(len(HAND_STREAM_IDX), HAND_DIM)
        self.body_proj = _branch(len(BODY_STREAM_IDX), BODY_DIM)
        self.face_proj = _branch(len(FACE_STREAM_IDX), FACE_DIM)
        self.fusion_norm = nn.LayerNorm(D_MODEL)
        self.pos_embed = nn.Parameter(torch.zeros(1, SEQ_LEN, D_MODEL))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.embed_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=FFN_DIM,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=N_LAYERS,
            norm=nn.LayerNorm(D_MODEL),
            enable_nested_tensor=False,
        )
        self.attn_pool = nn.Linear(D_MODEL, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(D_MODEL * 2),
            nn.Dropout(dropout),
            nn.Linear(D_MODEL * 2, num_classes),
        )

    def forward(self, x):
        tokens = torch.cat(
            [
                self.hand_proj(x.index_select(2, self.hand_idx)),
                self.body_proj(x.index_select(2, self.body_idx)),
                self.face_proj(x.index_select(2, self.face_idx)),
            ],
            dim=-1,
        )
        tokens = self.embed_dropout(self.fusion_norm(tokens) + self.pos_embed)
        features = self.encoder(tokens)
        weights = torch.softmax(self.attn_pool(features), dim=1)
        attended = torch.sum(features * weights, dim=1)
        return self.classifier(torch.cat([attended, features.mean(dim=1)], dim=1))


class LSC5_CNN1D(nn.Module):
    """CNN 1D compacta (~38k parámetros) con convoluciones depthwise-separable residuales."""

    def __init__(self, in_dim=INPUT_DIM, hidden=64, dropout=DROPOUT, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block1 = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, groups=hidden),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes),
        )

    def forward(self, x):
        # x: (B, T, D) -> (B, D, T)
        h = self.stem(x.transpose(1, 2))
        h = h + self.block1(h)
        h = h + self.block2(h)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        return self.head(pooled)


class LinearSegmentModel(nn.Module):
    def __init__(self, in_dim: int, num_classes=5, dropout=LIN_DROPOUT):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(self.drop(self.norm(x)))


# ========= Dataset y Augmentación de secuencias =========
class SequenceDataset(Dataset):
    def __init__(self, X_data, y_data, augment=False):
        self.X = torch.from_numpy(np.asarray(X_data, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y_data, dtype=np.int64))
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def _augment_sequence(self, x: np.ndarray) -> np.ndarray:
        if random.random() < 0.65:
            t = x.shape[0]
            gamma = random.uniform(0.90, 1.10)
            positions = (np.linspace(0, 1, t, dtype=np.float32) ** gamma) * (t - 1)
            lo = np.floor(positions).astype(np.int64)
            hi = np.minimum(lo + 1, t - 1)
            alpha = (positions - lo)[:, None]
            old_x = x
            x = x[lo] * (1.0 - alpha) + x[hi] * alpha
            near = np.clip(np.floor(positions + 0.5).astype(int), 0, t - 1)
            x[:, MASK_FEATURES] = old_x[near][:, MASK_FEATURES]

        if random.random() < 0.70:
            ang = math.radians(random.uniform(-10.0, 10.0))
            c, s = math.cos(ang), math.sin(ang)
            vx, vy = x[:, XY_X_IDX].copy(), x[:, XY_Y_IDX].copy()
            x[:, XY_X_IDX] = c * vx - s * vy
            x[:, XY_Y_IDX] = s * vx + c * vy

        if random.random() < 0.70:
            noise = np.random.normal(0, 0.015, size=x.shape).astype(np.float32)
            noise[:, MASK_FEATURES] = 0.0
            x = x + noise

        if random.random() < 0.25:
            drop_frames = np.random.rand(x.shape[0]) < 0.30
            if random.random() < 0.5:
                lh_cols = (
                    list(range(0, 63))
                    + list(range(126, 129))
                    + list(range(200, 263))
                    + list(range(326, 329))
                )
                lh_cols += list(range(400, 415)) + list(range(430, 433))
                x[np.ix_(drop_frames, lh_cols)] = 0.0
            else:
                rh_cols = (
                    list(range(63, 126))
                    + list(range(129, 132))
                    + list(range(263, 326))
                    + list(range(329, 332))
                )
                rh_cols += list(range(415, 430)) + list(range(433, 436))
                x[np.ix_(drop_frames, rh_cols)] = 0.0

        return x.astype(np.float32)

    def __getitem__(self, idx):
        x = self.X[idx].numpy().copy()
        if self.augment:
            x = self._augment_sequence(x)
        return torch.from_numpy(x), self.y[idx]


# ========= Entrenamiento y predicción genéricos para modelos de secuencia (Transformer y CNN1D) =========
def predict_seq_model(model, scaler, X_raw, X_mir, device, batch_size=64):
    model.eval()

    def _run(X_in):
        X_s = apply_sequence_scaler(X_in, scaler)
        loader = DataLoader(
            SequenceDataset(X_s, np.zeros(len(X_s), dtype=np.int64), augment=False),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        outs = []
        with torch.no_grad():
            for xb, _ in loader:
                logits = model(xb.to(device, non_blocking=True))
                outs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(outs, axis=0)

    p = _run(X_raw)
    if X_mir is not None:
        p = 0.5 * (p + _run(X_mir))
    return p


def fit_seq_model(
    model_cls,
    X_train_raw,
    y_train,
    X_valid_raw=None,
    y_valid=None,
    max_epochs=80,
    patience=15,
    seed=42,
    batch_size=32,
    device=torch.device("cpu"),
    num_workers=0,
    lr=LR,
):
    seed_everything(seed)
    scaler = fit_sequence_scaler(X_train_raw)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        SequenceDataset(
            apply_sequence_scaler(X_train_raw, scaler), y_train, augment=True
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    has_valid = X_valid_raw is not None and y_valid is not None
    if has_valid:
        valid_loader = DataLoader(
            SequenceDataset(
                apply_sequence_scaler(X_valid_raw, scaler), y_valid, augment=False
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
        )

    model = model_cls().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    amp_enabled = device.type == "cuda"
    try:
        grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_epoch, best_acc, best_loss = max_epochs, -1.0, float("inf")
    best_state, no_improve = None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                with torch.autocast(device_type="cuda", enabled=True):
                    loss = criterion(model(xb), yb)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        if has_valid:
            model.eval()
            val_loss_sum, val_seen, val_pred, val_true = 0.0, 0, [], []
            with torch.no_grad():
                for xb, yb in valid_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    val_loss_sum += float(loss.item()) * len(yb)
                    val_seen += len(yb)
                    val_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                    val_true.extend(yb.cpu().numpy().tolist())
            val_loss = val_loss_sum / max(val_seen, 1)
            val_acc = accuracy_score(val_true, val_pred)
            improved = (val_acc > best_acc + 1e-6) or (
                abs(val_acc - best_acc) <= 1e-6 and val_loss < best_loss
            )
            if improved:
                best_acc, best_loss, best_epoch = float(val_acc), float(val_loss), epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
        else:
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        scheduler.step()
        if has_valid and no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, int(best_epoch)


def run_loso_seq_model(
    model_cls, model_name, X, y, meta_df, X_mir, seed, args, device, lr=LR
):
    signers_arr = meta_df["signer"].to_numpy()
    signers = sorted(set(signers_arr))
    fold_rows, best_epochs = [], []
    oof_proba = np.full((len(y), len(CLASS_NAMES)), np.nan, dtype=np.float32)

    for fold_no, heldout in enumerate(signers, start=1):
        t0 = time.time()
        tr_idx = np.where(signers_arr != heldout)[0]
        va_idx = np.where(signers_arr == heldout)[0]

        inner_epochs = []
        for s_val in sorted(set(signers_arr[tr_idx])):
            itr = tr_idx[signers_arr[tr_idx] != s_val]
            iva = tr_idx[signers_arr[tr_idx] == s_val]
            Xi = np.concatenate([X[itr], X_mir[itr]])
            yi = np.concatenate([y[itr], y[itr]])
            _, _, be = fit_seq_model(
                model_cls,
                Xi,
                yi,
                X[iva],
                y[iva],
                max_epochs=args.max_epochs,
                patience=args.patience,
                seed=seed + fold_no,
                batch_size=args.batch_size,
                device=device,
                num_workers=args.num_workers,
                lr=lr,
            )
            inner_epochs.append(be)

        best_epoch = max(1, int(round(float(np.mean(inner_epochs)))))
        best_epochs.append(best_epoch)

        Xo = np.concatenate([X[tr_idx], X_mir[tr_idx]])
        yo = np.concatenate([y[tr_idx], y[tr_idx]])
        model, scaler, _ = fit_seq_model(
            model_cls,
            Xo,
            yo,
            None,
            None,
            max_epochs=best_epoch,
            patience=best_epoch,
            seed=seed + 100 + fold_no,
            batch_size=args.batch_size,
            device=device,
            num_workers=args.num_workers,
            lr=lr,
        )
        probs = predict_seq_model(model, scaler, X[va_idx], X_mir[va_idx], device)
        oof_proba[va_idx] = probs
        m = calculate_metrics(y[va_idx], probs.argmax(axis=1))
        fold_rows.append(
            {
                "model": model_name,
                "seed": seed,
                "heldout_signer": heldout,
                "best_epochs": best_epoch,
                **m,
            }
        )
        print(
            f"[{model_name} | seed={seed}] {heldout}: acc={m['accuracy']:.4f} | "
            f"f1={m['macro_f1']:.4f} | ep={best_epoch} | {time.time() - t0:.1f}s"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return fold_rows, oof_proba, best_epochs


# ========= Entrenamiento y predicción: Modelo Lineal por 4 tramos con rotación 2D =========
def fit_linear_segment(
    X_seq_tr, y_tr, seed, epochs, batch_size, device, rotate_2d=True
):
    seed_everything(seed)
    in_dim = X_seq_tr.shape[2] * (N_SEGMENTS + 1)
    ds = TensorDataset(torch.from_numpy(X_seq_tr), torch.from_numpy(y_tr))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    net = LinearSegmentModel(in_dim, len(CLASS_NAMES)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LIN_LR, weight_decay=LIN_WD)
    crit = nn.CrossEntropyLoss(label_smoothing=0.03)
    net.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            if rotate_2d:
                xb = batch_rotate_2d_torch(xb, max_deg=10.0, p=0.70)
            xvec = batch_seq_to_segment_vector_torch(xb, N_SEGMENTS)
            opt.zero_grad()
            crit(net(xvec), yb).backward()
            opt.step()
    net.eval()
    return net


def predict_linear_segment(net, X_vec, X_vec_mir, device):
    net.eval()
    with torch.no_grad():
        p1 = torch.softmax(net(torch.from_numpy(X_vec).to(device)), dim=1).cpu().numpy()
        p2 = (
            torch.softmax(net(torch.from_numpy(X_vec_mir).to(device)), dim=1)
            .cpu()
            .numpy()
        )
    return 0.5 * (p1 + p2)


def run_loso_linear(X, X_mir, X_vec, X_vec_mir, y, meta_df, seeds, args, device):
    signers_arr = meta_df["signer"].to_numpy()
    signers = sorted(set(signers_arr))
    oof_seeds = []
    for s in seeds:
        oof_p = np.zeros((len(y), len(CLASS_NAMES)), dtype=np.float32)
        for heldout in signers:
            tr = np.where(signers_arr != heldout)[0]
            va = np.where(signers_arr == heldout)[0]
            Xtr_seq = np.concatenate([X[tr], X_mir[tr]])
            ytr = np.concatenate([y[tr], y[tr]])
            net = fit_linear_segment(
                Xtr_seq,
                ytr,
                s,
                args.lin_epochs,
                args.batch_size,
                device,
                rotate_2d=True,
            )
            oof_p[va] = predict_linear_segment(net, X_vec[va], X_vec_mir[va], device)
        oof_seeds.append(oof_p)
    return np.mean(oof_seeds, axis=0)


# ========= Checkpoints y generación de submission.csv =========
def save_final_checkpoints(
    X, X_mir, y, final_ep_trans, final_ep_cnn, seeds, args, device, out_dir: Path
):
    X_all = np.concatenate([X, X_mir])
    y_all = np.concatenate([y, y])

    for s in seeds:
        t_net, t_scaler, _ = fit_seq_model(
            LSC5_Transformer,
            X_all,
            y_all,
            None,
            None,
            max_epochs=final_ep_trans,
            patience=final_ep_trans,
            seed=s + 999,
            batch_size=args.batch_size,
            device=device,
            num_workers=args.num_workers,
            lr=LR,
        )
        torch.save(
            {
                "model_state_dict": {
                    k: v.detach().cpu() for k, v in t_net.state_dict().items()
                },
                "scaler_mean": torch.as_tensor(t_scaler["mean"].copy()),
                "scaler_std": torch.as_tensor(t_scaler["std"].copy()),
                "seed": int(s),
                "final_epochs": int(final_ep_trans),
                "preprocess_version": PREPROCESS_VERSION,
            },
            out_dir / f"transformer_final_seed{s}.pt",
        )

        c_net, c_scaler, _ = fit_seq_model(
            LSC5_CNN1D,
            X_all,
            y_all,
            None,
            None,
            max_epochs=final_ep_cnn,
            patience=final_ep_cnn,
            seed=s + 999,
            batch_size=args.batch_size,
            device=device,
            num_workers=args.num_workers,
            lr=1e-3,
        )
        torch.save(
            {
                "model_state_dict": {
                    k: v.detach().cpu() for k, v in c_net.state_dict().items()
                },
                "scaler_mean": torch.as_tensor(c_scaler["mean"].copy()),
                "scaler_std": torch.as_tensor(c_scaler["std"].copy()),
                "seed": int(s),
                "final_epochs": int(final_ep_cnn),
                "preprocess_version": PREPROCESS_VERSION,
            },
            out_dir / f"cnn1d_final_seed{s}.pt",
        )

        l_net = fit_linear_segment(
            X_all,
            y_all,
            s + 999,
            args.lin_epochs,
            args.batch_size,
            device,
            rotate_2d=True,
        )
        torch.save(
            {
                "model_state_dict": {
                    k: v.detach().cpu() for k, v in l_net.state_dict().items()
                },
                "in_dim": int(X_all.shape[2] * (N_SEGMENTS + 1)),
                "seed": int(s),
                "lin_epochs": int(args.lin_epochs),
                "preprocess_version": PREPROCESS_VERSION,
            },
            out_dir / f"linear_final_seed{s}.pt",
        )
        print(
            f"Checkpoints finales (Transformer + CNN1D + Lineal) guardados para seed={s}"
        )


def run_test_inference(
    test_dir: Path, seeds, out_dir: Path, data_root: Path, template_path, device
):
    paths = sorted(Path(test_dir).rglob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No hay archivos .npy en {test_dir}")
    ids, feats, mirrored = [], [], []
    for p in tqdm(paths, desc="Procesando test"):
        raw = load_landmark_array(p)
        ids.append(p.stem)
        feats.append(full_features(raw, SEQ_LEN))
        mirrored.append(full_features(mirror_raw(raw), SEQ_LEN))
    Xt = np.stack(feats).astype(np.float32)
    Xm = np.stack(mirrored).astype(np.float32)
    Xt_vec = seq_to_segment_vector(Xt, N_SEGMENTS)
    Xm_vec = seq_to_segment_vector(Xm, N_SEGMENTS)

    t_probs, c_probs, l_probs = [], [], []
    for s in seeds:
        ckp_t = torch.load(
            out_dir / f"transformer_final_seed{s}.pt",
            map_location="cpu",
            weights_only=True,
        )
        t_net = LSC5_Transformer().to(device)
        t_net.load_state_dict(ckp_t["model_state_dict"])
        t_scaler = {
            "mean": ckp_t["scaler_mean"].numpy(),
            "std": ckp_t["scaler_std"].numpy(),
        }
        t_probs.append(predict_seq_model(t_net, t_scaler, Xt, Xm, device))

        ckp_c = torch.load(
            out_dir / f"cnn1d_final_seed{s}.pt", map_location="cpu", weights_only=True
        )
        c_net = LSC5_CNN1D().to(device)
        c_net.load_state_dict(ckp_c["model_state_dict"])
        c_scaler = {
            "mean": ckp_c["scaler_mean"].numpy(),
            "std": ckp_c["scaler_std"].numpy(),
        }
        c_probs.append(predict_seq_model(c_net, c_scaler, Xt, Xm, device))

        ckp_l = torch.load(
            out_dir / f"linear_final_seed{s}.pt", map_location="cpu", weights_only=True
        )
        l_net = LinearSegmentModel(ckp_l["in_dim"], len(CLASS_NAMES)).to(device)
        l_net.load_state_dict(ckp_l["model_state_dict"])
        l_probs.append(predict_linear_segment(l_net, Xt_vec, Xm_vec, device))

    P_test = (
        np.mean(t_probs, axis=0) + np.mean(c_probs, axis=0) + np.mean(l_probs, axis=0)
    ) / 3.0
    labels = [ID_TO_CLASS[int(i)] for i in P_test.argmax(axis=1)]
    sub = pd.DataFrame({"video_id": ids, "predicted_sign": labels})

    template_files = (
        [Path(template_path)]
        if template_path
        else list(data_root.rglob("submission_template.csv"))
    )
    if template_files and template_files[0].exists():
        tmpl = pd.read_csv(template_files[0])
        if set(tmpl["video_id"]) != set(sub["video_id"]):
            raise RuntimeError(
                "Los video_id de test y submission_template.csv no coinciden."
            )
        sub = tmpl[["video_id"]].merge(
            sub, on="video_id", how="left", validate="one_to_one"
        )

    sub_path = out_dir / "submission.csv"
    sub.to_csv(sub_path, index=False, encoding="utf-8")
    print(f"submission.csv generado en: {sub_path} ({len(sub)} filas)")
    print("Distribución predicha:", Counter(sub["predicted_sign"].tolist()))


# ========= Main =========
def parse_args():
    p = argparse.ArgumentParser(
        description="LSC5 T10 v2: Ensamble 3 modelos (Transformer + Lineal Rot2D + CNN1D)"
    )
    p.add_argument(
        "--data-root", type=str, default="./data", help="Carpeta raíz del dataset"
    )
    p.add_argument(
        "--train-dir", type=str, default=None, help="Ruta directa a train/landmarks"
    )
    p.add_argument(
        "--test-dir", type=str, default=None, help="Ruta directa a test/landmarks"
    )
    p.add_argument(
        "--template-path", type=str, default=None, help="Ruta a submission_template.csv"
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="./outputs/lsc5_t10_ens3",
        help="Directorio de salida",
    )
    p.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Dispositivo",
    )
    p.add_argument(
        "--threads", type=int, default=None, help="Hilos de CPU (ej. 8 en cronos)"
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Workers del DataLoader (0 recomendado en Windows/CPU)",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--lin-epochs", type=int, default=LIN_EPOCHS)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    p.add_argument(
        "--train-final",
        action="store_true",
        help="Entrena y guarda checkpoints finales con M02+M03+M04",
    )
    p.add_argument(
        "--run-test-inference",
        action="store_true",
        help="Genera submission.csv usando los checkpoints finales",
    )
    p.add_argument(
        "--use-codecarbon",
        action="store_true",
        help="Mide energía y CO2 con codecarbon (modo offline)",
    )
    p.add_argument(
        "--country-iso",
        type=str,
        default="CHL",
        help="Código ISO de país para CodeCarbon offline",
    )
    p.add_argument(
        "--quick-test",
        action="store_true",
        help="Prueba rápida (1 semilla, 2 épocas) solo para verificar código",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick_test:
        args.seeds = [42]
        args.max_epochs = 2
        args.patience = 2
        args.lin_epochs = 2

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Pediste --device cuda pero PyTorch no detecta GPU CUDA."
            )
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p_trans = sum(p.numel() for p in LSC5_Transformer().parameters())
    p_cnn = sum(p.numel() for p in LSC5_CNN1D().parameters())
    p_lin = sum(
        p.numel() for p in LinearSegmentModel(INPUT_DIM * (N_SEGMENTS + 1)).parameters()
    )

    print("=" * 72)
    print(f"PyTorch {torch.__version__} | Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"CPU threads: {torch.get_num_threads()}")
    print(
        f"Parámetros -> Transformer: {p_trans:,} | CNN1D: {p_cnn:,} | Lineal: {p_lin:,} | Total: {p_trans + p_cnn + p_lin:,}"
    )
    print(f"Semillas ({len(args.seeds)}): {args.seeds} | Salida: {out_dir}")
    print("=" * 72)

    tracker = None
    if args.use_codecarbon:
        try:
            from codecarbon import OfflineEmissionsTracker

            tracker = OfflineEmissionsTracker(
                country_iso_code=args.country_iso,
                output_dir=str(out_dir),
                output_file="emissions.csv",
                log_level="warning",
            )
            tracker.start()
            print(f"CodeCarbon iniciado (offline, país={args.country_iso}).")
        except ImportError:
            warnings.warn(
                "codecarbon no está instalado; se continúa sin medición de emisiones."
            )

    t_global = time.time()
    try:
        train_dir = discover_train_landmarks(Path(args.data_root), args.train_dir)
        records = [parse_sample_metadata(p) for p in sorted(train_dir.rglob("*.npy"))]
        metadata = pd.DataFrame(records)
        found_signers = sorted(metadata["signer"].unique().tolist())
        if found_signers != PUBLIC_SIGNERS_EXPECTED:
            raise RuntimeError(
                f"Esperaba {PUBLIC_SIGNERS_EXPECTED}, encontré {found_signers}."
            )
        print(f"Landmarks train: {train_dir} ({len(metadata)} muestras)")

        X = build_feature_tensor(metadata, mirror=False)
        X_mirror = build_feature_tensor(metadata, mirror=True)
        y = metadata["y"].to_numpy(dtype=np.int64)
        signers_arr = metadata["signer"].to_numpy()

        # 1. LOSO Transformer compacto
        t_rows, oof_t_by_seed, ep_t_by_seed = [], {}, {}
        for s in args.seeds:
            rows, probs, eps = run_loso_seq_model(
                LSC5_Transformer,
                "Transformer",
                X,
                y,
                metadata,
                X_mirror,
                s,
                args,
                device,
                lr=LR,
            )
            t_rows.extend(rows)
            oof_t_by_seed[s] = probs
            ep_t_by_seed[s] = eps

        P_trans = np.mean([oof_t_by_seed[s] for s in args.seeds], axis=0)
        pred_trans = P_trans.argmax(axis=1)
        m_trans = calculate_metrics(y, pred_trans)
        print(
            f"\n>>> Transformer OOF Accuracy: {m_trans['accuracy']:.6f} | Macro-F1: {m_trans['macro_f1']:.6f}"
        )
        for h in sorted(set(signers_arr)):
            mask = signers_arr == h
            print(f"    {h}: {accuracy_score(y[mask], pred_trans[mask]):.4f}")

        # 2. LOSO CNN1D compacta
        c_rows, oof_c_by_seed, ep_c_by_seed = [], {}, {}
        for s in args.seeds:
            rows, probs, eps = run_loso_seq_model(
                LSC5_CNN1D, "CNN1D", X, y, metadata, X_mirror, s, args, device, lr=1e-3
            )
            c_rows.extend(rows)
            oof_c_by_seed[s] = probs
            ep_c_by_seed[s] = eps

        P_cnn = np.mean([oof_c_by_seed[s] for s in args.seeds], axis=0)
        pred_cnn = P_cnn.argmax(axis=1)
        m_cnn = calculate_metrics(y, pred_cnn)
        print(
            f"\n>>> CNN1D OOF Accuracy: {m_cnn['accuracy']:.6f} | Macro-F1: {m_cnn['macro_f1']:.6f}"
        )
        for h in sorted(set(signers_arr)):
            mask = signers_arr == h
            print(f"    {h}: {accuracy_score(y[mask], pred_cnn[mask]):.4f}")

        # 3. LOSO Modelo Lineal por 4 tramos con rotación 2D
        X_vec = seq_to_segment_vector(X, N_SEGMENTS)
        X_vec_mir = seq_to_segment_vector(X_mirror, N_SEGMENTS)
        P_lin = run_loso_linear(
            X, X_mirror, X_vec, X_vec_mir, y, metadata, args.seeds, args, device
        )
        pred_lin = P_lin.argmax(axis=1)
        m_lin = calculate_metrics(y, pred_lin)
        print(
            f"\n>>> Lineal 4 tramos (Rot2D) OOF Accuracy: {m_lin['accuracy']:.6f} | Macro-F1: {m_lin['macro_f1']:.6f}"
        )
        for h in sorted(set(signers_arr)):
            mask = signers_arr == h
            print(f"    {h}: {accuracy_score(y[mask], pred_lin[mask]):.4f}")

        # 4. Comparación de Ensambles
        P_ens2 = 0.5 * P_trans + 0.5 * P_lin
        pred_ens2 = P_ens2.argmax(axis=1)
        m_ens2 = calculate_metrics(y, pred_ens2)
        print(
            f"\n>>> ENSAMBLE 50/50 (Transformer + Lineal) OOF Accuracy: {m_ens2['accuracy']:.6f} | Macro-F1: {m_ens2['macro_f1']:.6f} | Errores: {int((pred_ens2 != y).sum())}/{len(y)}"
        )
        for h in sorted(set(signers_arr)):
            mask = signers_arr == h
            print(f"    {h}: {accuracy_score(y[mask], pred_ens2[mask]):.4f}")

        P_ens3 = (P_trans + P_cnn + P_lin) / 3.0
        pred_ens3 = P_ens3.argmax(axis=1)
        m_ens3 = calculate_metrics(y, pred_ens3)
        print(
            f"\n>>> ENSAMBLE 3 MODELOS (1/3 Trans + 1/3 CNN1D + 1/3 Lineal) OOF Accuracy: {m_ens3['accuracy']:.6f} | Macro-F1: {m_ens3['macro_f1']:.6f}"
        )
        print(f"    Errores totales: {int((pred_ens3 != y).sum())} / {len(y)}")
        for h in sorted(set(signers_arr)):
            mask = signers_arr == h
            print(f"    {h}: {accuracy_score(y[mask], pred_ens3[mask]):.4f}")

        print(
            "\n"
            + classification_report(
                y, pred_ens3, target_names=CLASS_NAMES, digits=4, zero_division=0
            )
        )
        print_error_audit(
            metadata, y, pred_ens3, "Ensamble 3 modelos (1/3 + 1/3 + 1/3)"
        )

        # Guardar artefactos OOF
        pd.DataFrame(t_rows + c_rows).to_csv(
            out_dir / "loso_per_seed_fold.csv", index=False
        )
        np.save(out_dir / "oof_proba_transformer_mean.npy", P_trans)
        np.save(out_dir / "oof_proba_cnn1d_mean.npy", P_cnn)
        np.save(out_dir / "oof_proba_linear_mean.npy", P_lin)
        np.save(out_dir / "oof_proba_ensemble_50_50.npy", P_ens2)
        np.save(out_dir / "oof_proba_ensemble_3models.npy", P_ens3)

        final_ep_trans = max(
            1, int(np.median([e for s in args.seeds for e in ep_t_by_seed[s]]))
        )
        final_ep_cnn = max(
            1, int(np.median([e for s in args.seeds for e in ep_c_by_seed[s]]))
        )
        summary = {
            "preprocess_version": PREPROCESS_VERSION,
            "device": str(device),
            "seeds": args.seeds,
            "params": {
                "transformer": p_trans,
                "cnn1d": p_cnn,
                "linear": p_lin,
                "total": p_trans + p_cnn + p_lin,
            },
            "final_epochs_transformer": final_ep_trans,
            "final_epochs_cnn1d": final_ep_cnn,
            "lin_epochs": args.lin_epochs,
            "transformer_oof": m_trans,
            "cnn1d_oof": m_cnn,
            "linear_oof": m_lin,
            "ensemble_50_50_oof": m_ens2,
            "ensemble_3models_oof": m_ens3,
            "ensemble_3models_errors": int((pred_ens3 != y).sum()),
            "elapsed_seconds": round(time.time() - t_global, 2),
        }
        with open(out_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        if args.train_final:
            print(
                f"\nEntrenando modelos finales (Trans {final_ep_trans} ep, CNN1D {final_ep_cnn} ep, Lineal {args.lin_epochs} ep)..."
            )
            save_final_checkpoints(
                X,
                X_mirror,
                y,
                final_ep_trans,
                final_ep_cnn,
                args.seeds,
                args,
                device,
                out_dir,
            )

        if args.run_test_inference:
            if args.quick_test:
                raise RuntimeError("No generes submission.csv con --quick-test.")
            test_dir = discover_test_landmarks(Path(args.data_root), args.test_dir)
            run_test_inference(
                test_dir,
                args.seeds,
                out_dir,
                Path(args.data_root),
                args.template_path,
                device,
            )

        print(f"\nTodo finalizado en {(time.time() - t_global) / 60:.2f} minutos.")
    finally:
        if tracker is not None:
            emissions = tracker.stop()
            print(
                f"CodeCarbon finalizado. Emisiones estimadas: {emissions:.6f} kg CO2eq"
            )


if __name__ == "__main__":
    main()
