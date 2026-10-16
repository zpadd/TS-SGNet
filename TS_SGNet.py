import os
import gc
import json
import re
import math
import copy
import random
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

from scipy.interpolate import griddata
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch_geometric.utils import softmax as pyg_softmax

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False





os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


FEATURE_COLS = ["TOC", "RO", "ST", "SRH", "RE", "GT", "TE"]

WELL_FILE = "蠡县探井连续数据集.csv"
GRID_FILE = "蠡县网格连续数据集.csv"

GRID_RES = 200
RANDOM_SEED = 1965922
VAL_SIZE = 0.2

EPOCHS = 800
PATIENCE = 30
WARMUP_EPOCHS = 40




BEST_SELECTION_START_EPOCH = 80
EARLY_STOP_START_EPOCH = 300

LR = 2.0e-4
WEIGHT_DECAY = 3.0e-6
DROPOUT = 0.2



TRANSFORMER_D_MODEL = 48
TRANSFORMER_NHEAD = 4
TRANSFORMER_LAYERS = 1
TRANSFORMER_FF_DIM = 96
TRANSFORMER_CHUNK_SIZE = 512



MODEL_VARIANT = "transformer_cvae_ssn_gat_backend_u"
EXPERIMENT_ROADS = [
    {
        "variant": "transformer_cvae_ssn_gat_backend_u",
        "dirname": "Fused_front1_backend2_CAGrad_backend_u",
        "label": "Front file 1 + backend file 2 + CAGrad | backend input u",
    },
]


GAT_PRIOR_BETA = 0.80
GAT_EDGE_PRIOR_EPS = 1e-4

LOSS_NAME = "WeightedFocal_QQ_EMA_PseudoKeepCONS"
FOCAL_ALPHA = 0.50
FOCAL_GAMMA = 1.00



REAL_OIL_WEIGHT = 1.10
REAL_DRY_WEIGHT = 1.00
SSN_SCALES = [100, 400, 800]






CONS_WEIGHT = 0.0010
CONS_START_EPOCH = 80
CONS_RAMP_EPOCHS = 80
CONS_CONF_MARGIN = 0.12
CONS_MAX_SAMPLES = 12000
SMOOTH_WEIGHT = 0.0003
RECON_WEIGHT = 0.0005
KL_WEIGHT = 0.0001
CAGRAD_ALPHA = 0.50
CAGRAD_RESCALE = 1











PROJ_DIM = 32
PROJECTION_HEAD_HIDDEN_DIM = 64
PROJECTION_HEAD_DIM = 32

OUTPUT_DIRNAME = "LX_group001_weighted_FOCAL_QQ_EMA_PseudoKeepCONS_backend_u_PosNeg1to1_result"


SAVE_EPOCH_PREDICTION_MAPS = True
EPOCH_PREDICTION_DIRNAME = "每轮预测结果图"



RESERVE_PROB_THRESHOLD = 0.50




WORK_BOUNDARY_CANDIDATES = [
    "蠡县新工区边界.txt",
    "工区边界.txt",
    "work_boundary.txt",
    "boundary.txt",
    "工区范围.txt",
    "工区边界.csv",
    "work_boundary.csv",
    "boundary.csv",
    "工区范围.csv",
]

RESERVE_BOUNDARY_CANDIDATES = [
    "蠡县新工区储量.bln",
    "蠡县新工区储量.txt",
    "储量区.bln",
    "reserve.bln",
    "reserve.txt",
]


PSEUDO_CFG = {





    "stage2_start_epoch": 40,
    "stage2_end_epoch": 999999,
    "stage2_rebuild_each_epoch": True,
    "pseudo_strategy": "ema_quality_quantity_balance",


    "adaptive_threshold": True,
    "adaptive_threshold_stat": "freematch_style_ema_confidence",
    "ema_balance_enabled": True,
    "ema_momentum": 0.50,
    "ema_min_global_conf": 0.55,
    "ema_max_global_conf": 0.95,
    "ema_min_class_conf": 0.55,
    "ema_class_prior_blend": 0.35,   
    "ema_min_class_ratio": 0.10,
    "ema_max_class_ratio": 0.90,




    "quality_quantity_balance": True,
    "pseudo_quality_weight_min": 0.35,
    "pseudo_quality_power": 1.0,
    "use_uncertainty_gate": True,
    "ema_std_multiplier": 2.0,
    "ema_std_min": 0.02,
    "ema_std_max": 0.20,


    "stage2_total_topk_start": 200,
    "stage2_total_topk_mid": 400,
    "stage2_total_topk_end": 600,
    "allow_reallocate_unused_capacity": False,


    "softmatch_weighting": True,
    "softmatch_sigma_min": 1e-3,
    "csl_score_quantile": 0.0,
    "csl_min_conf_margin": 0.0,
    "csl_prob_weight": 0.0,
    "csl_conf_weight": 0.0,
    "oil_start": 0.80,
    "oil_mid": 0.80,
    "oil_end": 0.80,
    "dry_start": 0.20,
    "dry_mid": 0.20,
    "dry_end": 0.20,
    "stage2_oil_min_prob": 0.80,
    "stage2_dry_max_prob": 0.20,


    "std_max": 1.0,
    "dry_std_max": 1.0,
    "stage2_oil_std_max": 1.0,
    "stage2_dry_std_max": 1.0,


    "projection_sim_min": -999.0,
    "projection_margin_min": -999.0,
    "dry_projection_sim_min": -999.0,
    "dry_projection_margin_min": -999.0,
    "stage2_proj_oil_sim_min": -999.0,
    "stage2_proj_oil_margin_min": -999.0,
    "stage2_proj_dry_sim_min": -999.0,
    "stage2_proj_dry_margin_min": -999.0,
    "proj_oil_n_clusters": 0,
    "proj_dry_n_clusters": 0,
    "proj_score_weight": 0.0,
    "proj_margin_score_weight": 0.0,
    "use_projection_as_soft_score": False,
    "uncertainty_penalty": 0.0,



    "direct_topk_no_threshold": False,
    "rebuild_pseudo_pool_each_epoch": True,
    "target_pos_neg_ratio": 1.00,
    "target_pos_neg_low": 1.00,
    "target_pos_neg_hi": 1.00,
    "balance_pseudo_pool_capacity": True,
    "force_final_pos_neg_1to1": True,


    "oil_topk_max": 300,
    "dry_topk_max": 300,
    "stage2_topk_start": 300,
    "stage2_topk_mid": 300,
    "stage2_topk_end": 300,
    "stage2_oil_topk_start": 300,
    "stage2_oil_topk_mid": 300,
    "stage2_oil_topk_end": 300,
    "stage2_dry_topk_start": 300,
    "stage2_dry_topk_mid": 300,
    "stage2_dry_topk_end": 300,


    "pseudo_oil_weight": 0.30,
    "pseudo_dry_weight": 0.25,
    "pseudo_oil_label_min": 1.00,
    "pseudo_oil_label_max": 1.00,


    "dry_enable_min_ratio": 0.00,
    "dry_start_ratio": 0.00,
    "oil_threshold_floor": 0.55,
    "dry_threshold_ceiling": 0.45,
    "target_new_start": -1,
    "target_new_mid": -1,
    "target_new_end": -1,
    "oil_relax_enabled": False,
    "oil_relax_until_ratio": 0.00,
    "oil_relax_topk_when_below_ratio": 0,
    "oil_relax_threshold": 0.80,
    "oil_relax_std_max": 1.0,
    "oil_relax_min_prob": 0.80,
    "max_new_oil_early": 999999,
    "max_new_oil_mid": 999999,
    "max_new_oil_late": 999999,
    "max_new_dry_early": 999999,
    "max_new_dry_mid": 999999,
    "max_new_dry_late": 999999,
    "oil_keep_min_prob": 0.55,
    "dry_keep_max_prob": 0.45,
    "cap_start": 1.00,
    "cap_mid": 1.00,
    "cap_end": 1.00,
    "proj_oil_sim_min": -999.0,
    "proj_use_dry_margin": False,
    "proj_oil_margin": -999.0,
    "strict_keep_start_epoch": 20,
    "strict_oil_keep_prob": 0.55,
    "strict_dry_keep_prob": 0.45,
}








def seed_everything(seed: int = 2026):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.mkldnn.enabled = False
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass


def release_cuda_memory():
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def finite_tensor(
    x: torch.Tensor,
    nan: float = 0.0,
    pos: float = 1e4,
    neg: float = -1e4,
) -> torch.Tensor:
    return torch.nan_to_num(x, nan=nan, posinf=pos, neginf=neg)


def finite_scalar_loss(x: torch.Tensor, device=None) -> torch.Tensor:
    if torch.isfinite(x):
        return x

    if device is None:
        device = x.device

    return torch.zeros((), device=device, dtype=torch.float32)


def safe_float(x, default: float = float("nan")) -> float:
    try:
        if torch.is_tensor(x):
            if x.numel() == 0:
                return float(default)
            x = x.detach().float().mean().cpu().item()
        elif isinstance(x, np.ndarray):
            if x.size == 0:
                return float(default)
            x = float(np.nanmean(x))
        else:
            x = float(x)

        if not np.isfinite(x):
            return float(default)

        return float(x)
    except Exception:
        return float(default)


def metric_str(x, digits: int = 4) -> str:
    v = safe_float(x)

    if not np.isfinite(v):
        return "nan"

    return f"{v:.{digits}f}"


def compute_auc_fpr(labels, probs, threshold: float = 0.5):
    y_true = np.asarray(labels).reshape(-1).astype(int)
    y_prob = np.asarray(probs).reshape(-1).astype(float)

    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[valid]
    y_prob = y_prob[valid]

    if y_true.size == 0:
        return float("nan"), float("nan")

    if len(np.unique(y_true)) < 2:
        auc_value = float("nan")
    else:
        try:
            fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
            auc_value = float(auc(fpr_arr, tpr_arr))
        except Exception:
            auc_value = float("nan")

    y_pred = (y_prob >= threshold).astype(int)

    neg_mask = y_true == 0
    denom = int(np.sum(neg_mask))

    if denom <= 0:
        fpr_value = float("nan")
    else:
        fp = int(np.sum((y_pred == 1) & neg_mask))
        fpr_value = float(fp / max(denom, 1))

    return auc_value, fpr_value





def strip_numeric_suffix(filename: str) -> str:

    return re.sub(r"\(\d+\)(?=\.[^.]+$)", "", filename)


def find_data_file(current_dir: str, canonical_name: str) -> str:

    exact = os.path.join(current_dir, canonical_name)

    if os.path.exists(exact):
        return exact

    numbered_matches = []

    try:
        for fname in os.listdir(current_dir):
            if fname != canonical_name and strip_numeric_suffix(fname) == canonical_name:
                numbered_matches.append(fname)
    except Exception:
        numbered_matches = []

    extra_msg = ""
    if numbered_matches:
        extra_msg = (
            "\n检测到带数字后缀的同名文件："
            + ", ".join(numbered_matches)
            + "\n请重命名为标准文件名后再运行："
            + canonical_name
        )

    raise FileNotFoundError(
        f"未找到标准命名的数据文件：{canonical_name}\n"
        f"请放在目录：{current_dir}\n"
        f"注意：导入数据集文件名不能带 (51)、(38) 等数字后缀。"
        f"{extra_msg}"
    )


def read_csv_auto(path: str) -> pd.DataFrame:
    last_err = None

    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e

    raise last_err


def normalize_xy_columns(df: pd.DataFrame) -> pd.DataFrame:

    rename = {}

    if "X" in df.columns and "x" not in df.columns:
        rename["X"] = "x"
    if "Y" in df.columns and "y" not in df.columns:
        rename["Y"] = "y"

    return df.rename(columns=rename)


def read_xy_points_auto(path: str) -> np.ndarray:

    number_pattern = re.compile(
        r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    )

    last_err = None






    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            pts = []

            with open(path, "r", encoding=enc, errors="ignore") as f:
                for line in f:
                    s = line.strip()

                    if not s:
                        continue
                    if s.startswith("#") or s.startswith("//"):
                        continue

                    nums = number_pattern.findall(s)

                    if len(nums) < 2:
                        continue

                    x = float(nums[0])
                    y = float(nums[1])


                    if len(nums) == 2 and abs(x) < 100000.0 and abs(y) <= 10.0:
                        continue

                    pts.append([x, y])

            if len(pts) >= 3:
                xy = np.asarray(pts, dtype=float)



                if xy.shape[0] >= 4:
                    med_x = float(np.nanmedian(xy[1:, 0]))
                    med_y = float(np.nanmedian(xy[1:, 1]))
                    first_is_header = (
                        abs(xy[0, 0]) < max(abs(med_x) * 0.01, 1e5)
                        and abs(xy[0, 1]) < max(abs(med_y) * 0.01, 1e5)
                    )
                    if first_is_header:
                        xy = xy[1:]


                valid = np.isfinite(xy).all(axis=1)
                xy = xy[valid]

                if xy.shape[0] >= 3:
                    return xy
        except Exception as e:
            last_err = e


    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            raw = pd.read_csv(path, encoding=enc)
            raw = normalize_xy_columns(raw)

            if "x" in raw.columns and "y" in raw.columns:
                xy = raw[["x", "y"]].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)
                if xy.shape[0] >= 3:
                    return xy
        except Exception as e:
            last_err = e

    raise ValueError(f"无法读取边界坐标文件：{path} | {last_err}")


def read_bln_segments_auto(path: str):

    number_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    segments = []

    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            segments = []
            current = []

            with open(path, "r", encoding=enc, errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    if s.startswith("#") or s.startswith("//"):
                        continue

                    nums = number_pattern.findall(s)
                    if len(nums) < 2:
                        continue

                    x = float(nums[0])
                    y = float(nums[1])



                    is_bln_header = (
                        len(nums) == 2
                        and abs(x) < 100000.0
                        and abs(y) <= 10.0
                    )

                    if is_bln_header:
                        if len(current) >= 2:
                            arr = np.asarray(current, dtype=float)
                            arr = arr[np.isfinite(arr).all(axis=1)]
                            if arr.shape[0] >= 2:
                                segments.append(arr)
                        current = []
                        continue

                    current.append([x, y])

            if len(current) >= 2:
                arr = np.asarray(current, dtype=float)
                arr = arr[np.isfinite(arr).all(axis=1)]
                if arr.shape[0] >= 2:
                    segments.append(arr)

            cleaned = []
            for arr in segments:
                if arr.shape[0] >= 2:

                    if arr.shape[0] >= 3 and np.linalg.norm(arr[0] - arr[-1]) > 1e-8:
                        arr = np.vstack([arr, arr[0]])
                    cleaned.append(arr)

            if len(cleaned) > 0:
                return cleaned
        except Exception:
            continue


    xy = read_xy_points_auto(path)
    if xy.shape[0] >= 3 and np.linalg.norm(xy[0] - xy[-1]) > 1e-8:
        xy = np.vstack([xy, xy[0]])
    return [xy]


def boundary_segments(boundary_obj):

    if boundary_obj is None:
        return []

    if isinstance(boundary_obj, (list, tuple)):
        out = []
        for seg in boundary_obj:
            arr = np.asarray(seg, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 2:
                out.append(arr[:, :2])
        return out

    arr = np.asarray(boundary_obj, dtype=float)
    if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 2:
        return [arr[:, :2]]

    return []


def has_boundary(boundary_obj, min_points: int = 3) -> bool:
    return any(seg.shape[0] >= min_points for seg in boundary_segments(boundary_obj))


def plot_boundary_segments(ax, boundary_obj, color: str, linewidth: float, label: str, zorder: int):
    segs = boundary_segments(boundary_obj)
    first = True
    for seg in segs:
        if seg.shape[0] < 2:
            continue
        ax.plot(
            seg[:, 0],
            seg[:, 1],
            color=color,
            linewidth=linewidth,
            label=label if first else None,
            zorder=zorder,
        )
        first = False


def boundary_union_mask(x_grid: np.ndarray, y_grid: np.ndarray, boundary_obj) -> np.ndarray:
    mask = np.zeros_like(x_grid, dtype=bool)
    for seg in boundary_segments(boundary_obj):
        if seg.shape[0] >= 3:
            try:
                mask |= polygon_to_mask(x_grid, y_grid, seg)
            except Exception:
                pass
    return mask

def read_optional_work_boundary(current_dir: str):

    candidate_paths = [os.path.join(current_dir, name) for name in WORK_BOUNDARY_CANDIDATES]

    seen = set()

    for path in candidate_paths:
        if path in seen:
            continue
        seen.add(path)

        if not os.path.exists(path):
            continue

        try:
            xy = read_xy_points_auto(path)

            if xy.shape[0] < 3:
                print(f"⚠️ 工区边界点数不足 3 个，已跳过：{path}")
                continue

            if np.linalg.norm(xy[0] - xy[-1]) > 1e-8:
                xy = np.vstack([xy, xy[0]])

            print(f"📌 已读取工区边界：{os.path.basename(path)} | 点数={xy.shape[0]}")
            return xy
        except Exception as e:
            print(f"⚠️ 工区边界文件读取失败，已跳过：{path} | {e}")

    numbered_boundary_files = []
    try:
        canonical_set = set(WORK_BOUNDARY_CANDIDATES)
        for fname in os.listdir(current_dir):
            if fname not in canonical_set and strip_numeric_suffix(fname) in canonical_set:
                numbered_boundary_files.append(fname)
    except Exception:
        numbered_boundary_files = []

    if numbered_boundary_files:
        print(
            "⚠️ 检测到带数字后缀的边界文件："
            + ", ".join(numbered_boundary_files)
            + "；代码不会读取它。请重命名为：蠡县新工区边界.txt"
        )

    print("📌 未发现标准命名工区边界文件：将根据网格有效范围推断工区边界；若网格全为有效值，则使用网格矩形范围。")
    return None


def read_optional_reserve_boundary(current_dir: str):

    candidate_paths = [os.path.join(current_dir, name) for name in RESERVE_BOUNDARY_CANDIDATES]

    seen = set()

    for path in candidate_paths:
        if path in seen:
            continue
        seen.add(path)

        if not os.path.exists(path):
            continue

        try:
            segs = read_bln_segments_auto(path)
            point_count = int(sum(seg.shape[0] for seg in segs))

            if len(segs) == 0 or point_count < 3:
                print(f"⚠️ 储量区边界点数不足 3 个，已跳过：{path}")
                continue

            print(
                f"📌 已读取储量区边界：{os.path.basename(path)} | "
                f"段数={len(segs)} | 点数={point_count}"
            )
            return segs
        except Exception as e:
            print(f"⚠️ 储量区边界文件读取失败，已跳过：{path} | {e}")

    numbered_reserve_files = []
    try:
        canonical_set = set(RESERVE_BOUNDARY_CANDIDATES)
        for fname in os.listdir(current_dir):
            if fname not in canonical_set and strip_numeric_suffix(fname) in canonical_set:
                numbered_reserve_files.append(fname)
    except Exception:
        numbered_reserve_files = []

    if numbered_reserve_files:
        print(
            "⚠️ 检测到带数字后缀的储量区文件："
            + ", ".join(numbered_reserve_files)
            + "；代码不会读取它。请重命名为：蠡县新工区储量.bln"
        )
    else:
        print("📌 未发现标准命名储量区文件：最终图不会绘制蓝色储量区边界。")

    return None


def polygon_to_mask(x_grid: np.ndarray, y_grid: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    points = np.column_stack([x_grid.reshape(-1), y_grid.reshape(-1)])
    mask = MplPath(polygon_xy[:, :2]).contains_points(points, radius=1e-9)
    return mask.reshape(x_grid.shape).astype(bool)


def infer_grid_same_order(grid_df: pd.DataFrame, x_grid: np.ndarray, y_grid: np.ndarray) -> bool:
    if "x" not in grid_df.columns or "y" not in grid_df.columns:
        return False

    gx = pd.to_numeric(grid_df["x"], errors="coerce").to_numpy(dtype=float)
    gy = pd.to_numeric(grid_df["y"], errors="coerce").to_numpy(dtype=float)

    target_x = np.asarray(x_grid, dtype=float).reshape(-1)
    target_y = np.asarray(y_grid, dtype=float).reshape(-1)

    return (
        len(grid_df) == target_x.size
        and np.all(np.isfinite(gx))
        and np.all(np.isfinite(gy))
        and np.allclose(gx, target_x, rtol=0.0, atol=1e-6)
        and np.allclose(gy, target_y, rtol=0.0, atol=1e-6)
    )


def values_to_grid_mask(
    grid_df: pd.DataFrame,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:

    values = np.asarray(values).reshape(-1)

    if infer_grid_same_order(grid_df, x_grid, y_grid):
        return (values.reshape(x_grid.shape) > 0.5).astype(bool)

    gx = pd.to_numeric(grid_df["x"], errors="coerce").to_numpy(dtype=float)
    gy = pd.to_numeric(grid_df["y"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(gx) & np.isfinite(gy) & np.isfinite(values)

    if np.sum(valid) == 0:
        return np.ones_like(x_grid, dtype=bool)

    target_points = np.column_stack([x_grid.reshape(-1), y_grid.reshape(-1)])
    nearest = griddata(
        np.column_stack([gx[valid], gy[valid]]),
        values[valid].astype(float),
        target_points,
        method="nearest",
    )

    nearest = np.nan_to_num(nearest, nan=0.0, posinf=0.0, neginf=0.0)
    return (nearest.reshape(x_grid.shape) > 0.5).astype(bool)


def infer_work_area_mask(
    grid_df: pd.DataFrame,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    feature_cols,
    boundary_xy=None,
) -> np.ndarray:

    if boundary_xy is not None:
        mask = polygon_to_mask(x_grid, y_grid, boundary_xy)
        if np.any(mask):
            return mask.astype(bool)

    mask_cols = [
        "inside",
        "mask",
        "valid",
        "work_area",
        "area_mask",
        "工区",
        "工区内",
        "有效区",
        "有效范围",
    ]

    for col in mask_cols:
        if col in grid_df.columns:
            raw = grid_df[col]

            if raw.dtype == bool:
                vals = raw.astype(float).to_numpy()
            else:
                vals = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)

            if np.isfinite(vals).sum() > 0:
                mask = values_to_grid_mask(grid_df, x_grid, y_grid, vals)
                if np.any(mask):
                    print(f"📌 已使用网格列 `{col}` 作为工区内部掩膜。")
                    return mask.astype(bool)

    gx = pd.to_numeric(grid_df["x"], errors="coerce").to_numpy(dtype=float)
    gy = pd.to_numeric(grid_df["y"], errors="coerce").to_numpy(dtype=float)

    feat_valid_list = []
    for col in feature_cols:
        if col in grid_df.columns:
            feat_valid_list.append(pd.to_numeric(grid_df[col], errors="coerce").to_numpy(dtype=float))

    if len(feat_valid_list) > 0:
        feat_arr = np.column_stack(feat_valid_list)
        valid_feature_row = np.any(np.isfinite(feat_arr), axis=1)
        valid_xy = np.isfinite(gx) & np.isfinite(gy)
        valid_values = (valid_xy & valid_feature_row).astype(float)

        if np.sum(valid_values > 0.5) >= 3 and np.sum(valid_values > 0.5) < len(valid_values):
            mask = values_to_grid_mask(grid_df, x_grid, y_grid, valid_values)
            if np.any(mask):
                print("📌 已根据网格特征非空区域推断工区内部掩膜。")
                return mask.astype(bool)

    return np.ones_like(x_grid, dtype=bool)


def mask_points_by_nearest_grid(
    xs,
    ys,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    xi = np.asarray(x_grid[0, :], dtype=float)
    yi = np.asarray(y_grid[:, 0], dtype=float)

    ix = np.searchsorted(xi, xs)
    iy = np.searchsorted(yi, ys)

    ix = np.clip(ix, 0, len(xi) - 1)
    iy = np.clip(iy, 0, len(yi) - 1)

    ix_left = np.clip(ix - 1, 0, len(xi) - 1)
    iy_down = np.clip(iy - 1, 0, len(yi) - 1)

    ix = np.where(np.abs(xi[ix_left] - xs) < np.abs(xi[ix] - xs), ix_left, ix)
    iy = np.where(np.abs(yi[iy_down] - ys) < np.abs(yi[iy] - ys), iy_down, iy)

    return mask[iy, ix].astype(bool)


def draw_work_area_boundary(
    ax,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    work_area_mask: np.ndarray,
    boundary_xy=None,
):

    if boundary_xy is not None and len(boundary_xy) >= 3:
        ax.plot(
            boundary_xy[:, 0],
            boundary_xy[:, 1],
            color="black",
            linewidth=1.8,
            label="工区边界",
            zorder=10,
        )
        return

    if np.any(work_area_mask) and not np.all(work_area_mask):
        try:
            ax.contour(
                x_grid,
                y_grid,
                work_area_mask.astype(float),
                levels=[0.5],
                colors="black",
                linewidths=1.8,
                zorder=10,
            )
            ax.plot([], [], color="black", linewidth=1.8, label="工区边界")
            return
        except Exception:
            pass

    xmin, xmax = float(x_grid.min()), float(x_grid.max())
    ymin, ymax = float(y_grid.min()), float(y_grid.max())
    ax.plot(
        [xmin, xmax, xmax, xmin, xmin],
        [ymin, ymin, ymax, ymax, ymin],
        color="black",
        linewidth=1.8,
        label="工区边界",
        zorder=10,
    )


def build_pseudo_explanation(best_row: Dict, final_row: Dict, cfg: Dict) -> str:
    best_row = best_row or {}
    final_row = final_row or {}

    best_epoch = int(best_row.get("epoch", -1))
    best_oil_pool = int(best_row.get("oil_pool", 0))
    best_dry_pool = int(best_row.get("dry_pool", 0))
    best_raw_oil = int(best_row.get("raw_oil", 0))
    best_raw_dry = int(best_row.get("raw_dry", 0))
    best_oil_th = float(best_row.get("oil_threshold", float("nan")))
    best_dry_th = float(best_row.get("dry_threshold", float("nan")))
    best_q = float(best_row.get("pseudo_quality_mean", float("nan")))
    best_target_oil_ratio = float(best_row.get("target_oil_ratio", float("nan")))

    final_epoch = int(final_row.get("epoch", -1))
    final_oil_pool = int(final_row.get("oil_pool", 0))
    final_dry_pool = int(final_row.get("dry_pool", 0))

    pseudo_start_epoch = int(cfg.get("stage2_start_epoch", WARMUP_EPOCHS))
    total_start = int(cfg.get("stage2_total_topk_start", cfg.get("stage2_topk_start", 300)))
    total_mid = int(cfg.get("stage2_total_topk_mid", cfg.get("stage2_topk_mid", total_start)))
    total_end = int(cfg.get("stage2_total_topk_end", cfg.get("stage2_topk_end", total_mid)))

    lines = []
    lines.append("伪标签筛选说明：")
    lines.append(
        "1. 当前伪标签采用 EMA quality-quantity balance 机制：使用 EMA 平滑后的动态阈值，不再使用固定 0.80/0.20 阈值；最终训练 Pos/Neg 保持 1:1。"
    )
    lines.append(
        f"2. 伪标签启动轮次为 Epoch={pseudo_start_epoch}；总容量随训练阶段递增：{total_start}->{total_mid}->{total_end}，并在高质量候选内按最终 Pos/Neg=1:1 裁剪。"
    )
    lines.append(
        "3. 质量控制：使用 EMA 平滑后的无标签置信度估计全局学习状态，并结合类别 EMA 置信度形成油/干类别自适应阈值；超过阈值越多，伪标签样本权重越高。"
    )
    lines.append(
        "4. 数量控制：根据真实井正负数量差和高质量伪标签候选数量，优先选择能够使真实+伪标签训练集 Pos/Neg 接近 1:1 的油/干伪标签组合。"
    )
    lines.append(
        f"5. 最优轮次 Epoch={best_epoch}，固定阈值候选已替换为动态候选：Raw(O/D)={best_raw_oil}/{best_raw_dry}，"
        f"Keep(O/D)={best_oil_pool}/{best_dry_pool}。"
    )
    if np.isfinite(best_oil_th) or np.isfinite(best_dry_th):
        lines.append(
            f"6. 最优轮次 EMA 动态概率阈值：OilMin={best_oil_th:.3f}、DryMax={best_dry_th:.3f}；"
            f"当前目标油类比例={best_target_oil_ratio:.3f}，平均伪标签质量权重={best_q:.3f}。"
        )

    if best_epoch >= 0 and best_epoch < pseudo_start_epoch:
        lines.append(
            f"7. 最优轮次早于伪标签启动轮次 Epoch={pseudo_start_epoch}，因此该轮只使用真实井强监督和辅助损失训练。"
        )
    elif best_oil_pool == 0 and best_dry_pool == 0:
        lines.append("7. 最优轮次没有无标签网格点通过 EMA 质量-数量筛选，因此最终图件只绘制真实井点。")
    else:
        lines.append("7. 伪标签是无标签网格训练样本，不等同于实钻井位；最终图件仍只标注真实井点。")

    if final_epoch >= 0 and final_epoch != best_epoch:
        lines.append(
            f"8. 训练停止时最后一轮 Epoch={final_epoch} 的训练池 OilPool={final_oil_pool}、DryPool={final_dry_pool}；"
            "若与最优轮次不同，是因为最终预测恢复了 best_model_state，而不是最后一轮模型。"
        )

    return "\n".join(lines)


def get_maps(
    grid_df: pd.DataFrame,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    feature_cols,
) -> np.ndarray:
    if grid_df is None or len(grid_df) == 0:
        raise ValueError("grid_df 为空，无法构造网格特征。")

    if "x" not in grid_df.columns or "y" not in grid_df.columns:
        raise ValueError("grid_df 必须包含 x、y 坐标列。")

    target_x = np.asarray(x_grid, dtype=float).reshape(-1)
    target_y = np.asarray(y_grid, dtype=float).reshape(-1)
    target_points = np.column_stack([target_x, target_y])

    n_target = target_x.size

    out_features = []

    gx = pd.to_numeric(grid_df["x"], errors="coerce").to_numpy(dtype=float)
    gy = pd.to_numeric(grid_df["y"], errors="coerce").to_numpy(dtype=float)

    same_order = (
        len(grid_df) == n_target
        and np.all(np.isfinite(gx))
        and np.all(np.isfinite(gy))
        and np.allclose(gx, target_x, rtol=0.0, atol=1e-6)
        and np.allclose(gy, target_y, rtol=0.0, atol=1e-6)
    )

    xi = np.asarray(x_grid[0, :], dtype=float)
    yi = np.asarray(y_grid[:, 0], dtype=float)

    for col in feature_cols:
        if col not in grid_df.columns:
            raise ValueError(f"网格数据缺少特征列：{col}")

        vals = pd.to_numeric(grid_df[col], errors="coerce").to_numpy(dtype=float)

        if same_order:
            feat = vals.reshape(-1)

            finite = np.isfinite(feat)
            fill_value = float(np.nanmedian(feat[finite])) if np.any(finite) else 0.0

            feat = np.nan_to_num(
                feat,
                nan=fill_value,
                posinf=fill_value,
                neginf=fill_value,
            )

            out_features.append(feat.astype(np.float32))
            continue

        feat = None

        try:
            tmp = pd.DataFrame(
                {
                    "x": gx,
                    "y": gy,
                    "v": vals,
                }
            )

            tmp = tmp.dropna(subset=["x", "y"])

            pv = tmp.pivot_table(
                index="y",
                columns="x",
                values="v",
                aggfunc="mean",
            )

            pv = pv.sort_index(axis=0).sort_index(axis=1)

            if pv.shape == x_grid.shape:
                arr = pv.to_numpy(dtype=float)
                pv_y = pv.index.to_numpy(dtype=float)
                pv_x = pv.columns.to_numpy(dtype=float)

                if np.allclose(pv_y, yi, rtol=0.0, atol=1e-6) and np.allclose(pv_x, xi, rtol=0.0, atol=1e-6):
                    feat = arr.reshape(-1)
                elif np.allclose(pv_y[::-1], yi, rtol=0.0, atol=1e-6) and np.allclose(pv_x, xi, rtol=0.0, atol=1e-6):
                    feat = arr[::-1, :].reshape(-1)

        except Exception:
            feat = None

        if feat is None:
            points = np.column_stack([gx, gy])
            valid = np.isfinite(points).all(axis=1) & np.isfinite(vals)

            if np.sum(valid) == 0:
                feat = np.zeros(n_target, dtype=float)
            else:
                pts = points[valid]
                v = vals[valid]

                try:
                    linear = griddata(pts, v, target_points, method="linear")
                except Exception:
                    linear = np.full(n_target, np.nan, dtype=float)

                try:
                    nearest = griddata(pts, v, target_points, method="nearest")
                except Exception:
                    nearest = np.full(
                        n_target,
                        np.nanmedian(v) if np.any(np.isfinite(v)) else 0.0,
                        dtype=float,
                    )

                feat = np.where(np.isfinite(linear), linear, nearest)

        finite = np.isfinite(feat)
        fill_value = float(np.nanmedian(feat[finite])) if np.any(finite) else 0.0

        feat = np.nan_to_num(
            feat,
            nan=fill_value,
            posinf=fill_value,
            neginf=fill_value,
        ).astype(np.float32)

        out_features.append(feat)

    return np.stack(out_features, axis=1).astype(np.float32)


def prepare_data(current_dir: str, device):
    well_path = find_data_file(current_dir, WELL_FILE)
    grid_path = find_data_file(current_dir, GRID_FILE)

    well_df = read_csv_auto(well_path)
    grid_df = read_csv_auto(grid_path)

    if "X" not in well_df.columns or "Y" not in well_df.columns:
        raise ValueError("探井数据必须包含 X、Y 坐标列。")

    if "X" in grid_df.columns and "Y" in grid_df.columns:
        grid_df = grid_df.rename(columns={"X": "x", "Y": "y"})

    if "x" not in grid_df.columns or "y" not in grid_df.columns:
        raise ValueError("网格数据必须包含 X/Y 或 x/y 坐标列。")

    missing_well_cols = [c for c in FEATURE_COLS if c not in well_df.columns]
    missing_grid_cols = [c for c in FEATURE_COLS if c not in grid_df.columns]

    if missing_well_cols:
        raise ValueError(f"探井数据缺少特征列：{missing_well_cols}")

    if missing_grid_cols:
        raise ValueError(f"网格数据缺少特征列：{missing_grid_cols}")

    if "CLASS" in well_df.columns:
        well_df["label"] = well_df["CLASS"].astype(int)
    elif "label" in well_df.columns:
        well_df["label"] = well_df["label"].astype(int)
    else:
        raise ValueError("探井数据必须包含 CLASS 或 label 标签列，油井=1，干井=0。")

    bounds = [
        float(grid_df["x"].min()),
        float(grid_df["x"].max()),
        float(grid_df["y"].min()),
        float(grid_df["y"].max()),
    ]

    xi = np.linspace(bounds[0], bounds[1], GRID_RES)
    yi = np.linspace(bounds[2], bounds[3], GRID_RES)

    x_grid, y_grid = np.meshgrid(xi, yi)

    work_boundary_xy = read_optional_work_boundary(current_dir)
    reserve_boundary_xy = read_optional_reserve_boundary(current_dir)

    work_area_mask = infer_work_area_mask(
        grid_df=grid_df,
        x_grid=x_grid,
        y_grid=y_grid,
        feature_cols=FEATURE_COLS,
        boundary_xy=work_boundary_xy,
    )

    raw_feats = get_maps(grid_df, x_grid, y_grid, FEATURE_COLS)

    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(raw_feats)

    scaled_feats = np.nan_to_num(
        scaled_feats,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    x_data = torch.FloatTensor(scaled_feats).to(device)

    x_norm = (x_grid - bounds[0]) / max(bounds[1] - bounds[0], 1e-8)
    y_norm = (y_grid - bounds[2]) / max(bounds[3] - bounds[2], 1e-8)

    coords_np = np.stack(
        [
            x_norm.flatten(),
            y_norm.flatten(),
        ],
        axis=-1,
    ).astype(np.float32)

    coords = torch.FloatTensor(coords_np).to(device)

    orig_indices = np.array(
        [
            np.argmin(np.abs(yi - wy)) * GRID_RES + np.argmin(np.abs(xi - wx))
            for wx, wy in zip(well_df["X"].values, well_df["Y"].values)
        ],
        dtype=np.int64,
    )

    orig_labels = well_df["label"].values.astype(np.int64)

    print(f"📂 数据目录：{current_dir}")
    print(f"📌 探井文件：{os.path.basename(well_path)}")
    print(f"📌 网格文件：{os.path.basename(grid_path)}")
    print(f"📌 特征列：{FEATURE_COLS}")
    print(f"📌 网格分辨率：{GRID_RES} | 网格数：{GRID_RES * GRID_RES}")
    print(
        f"📌 井点数：{len(orig_labels)} | "
        f"油井：{int(np.sum(orig_labels == 1))} | "
        f"干井：{int(np.sum(orig_labels == 0))}"
    )

    return {
        "current_dir": current_dir,
        "feature_cols": FEATURE_COLS,
        "feat_dim": len(FEATURE_COLS),
        "bounds": bounds,
        "grid_res": GRID_RES,
        "well_df": well_df,
        "grid_df": grid_df,
        "xi": xi,
        "yi": yi,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "work_boundary_xy": work_boundary_xy,
        "reserve_boundary_xy": reserve_boundary_xy,
        "work_area_mask": work_area_mask,
        "reserve_prob_threshold": RESERVE_PROB_THRESHOLD,
        "x_data": x_data,
        "coords": coords,
        "orig_indices": orig_indices,
        "orig_labels": orig_labels,
    }





class FeatureTokenTransformer(nn.Module):

    def __init__(
        self,
        feat_dim: int,
        d_model: int = TRANSFORMER_D_MODEL,
        nhead: int = TRANSFORMER_NHEAD,
        num_layers: int = TRANSFORMER_LAYERS,
        ff_dim: int = TRANSFORMER_FF_DIM,
        dropout_rate: float = DROPOUT,
        chunk_size: int = TRANSFORMER_CHUNK_SIZE,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.seq_len = self.feat_dim
        self.d_model = int(d_model)
        self.chunk_size = int(max(1, chunk_size))

        self.scalar_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.token_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.token_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def _encode_chunk(self, scalars: torch.Tensor, inject_noise: bool) -> torch.Tensor:
        tokens = self.scalar_proj(scalars.unsqueeze(-1))
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.token_embed[:, : tokens.size(1), :]
        if inject_noise and self.training:
            tokens = tokens + torch.randn_like(tokens) * 0.01
        if self.training:
            h = checkpoint(self.encoder, tokens, use_reentrant=False)
        else:
            h = self.encoder(tokens)
        pooled = self.out_norm(h[:, 0, :])
        return finite_tensor(pooled, nan=0.0, pos=10.0, neg=-10.0)

    def forward(self, x_feat: torch.Tensor, inject_noise: bool = False) -> torch.Tensor:
        x_feat = finite_tensor(x_feat, nan=0.0, pos=8.0, neg=-8.0).float()
        outs = []
        for start in range(0, x_feat.size(0), self.chunk_size):
            end = min(start + self.chunk_size, x_feat.size(0))
            outs.append(self._encode_chunk(x_feat[start:end], inject_noise=inject_noise))
        return torch.cat(outs, dim=0)


class CoordCVAE(nn.Module):

    def __init__(
        self,
        feat_dim: int,
        d_model: int = TRANSFORMER_D_MODEL,
        coord_dim: int = 2,
        hidden: int = 128,
        latent: int = PROJ_DIM,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.transformer_encoder = FeatureTokenTransformer(
            feat_dim=feat_dim,
            d_model=d_model,
            nhead=TRANSFORMER_NHEAD,
            num_layers=TRANSFORMER_LAYERS,
            ff_dim=TRANSFORMER_FF_DIM,
            dropout_rate=DROPOUT,
            chunk_size=TRANSFORMER_CHUNK_SIZE,
        )
        self.encoder_mlp = nn.Sequential(
            nn.Linear(d_model + coord_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.fc_u = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent + coord_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def reparameterize(self, u: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return u + eps * std
        return u

    def forward(self, x_feat: torch.Tensor, coords: torch.Tensor, inject_noise: bool = False) -> Dict[str, torch.Tensor]:
        x_feat = finite_tensor(x_feat, nan=0.0, pos=8.0, neg=-8.0).float()
        coords = finite_tensor(coords, nan=0.0, pos=1.0, neg=0.0).float().clamp(0.0, 1.0)

        transformer_feat = self.transformer_encoder(x_feat, inject_noise=inject_noise)
        transformer_feat = finite_tensor(transformer_feat, nan=0.0, pos=10.0, neg=-10.0)

        enc_in = torch.cat([transformer_feat, coords], dim=1)
        h = self.encoder_mlp(enc_in)
        h = finite_tensor(h, nan=0.0, pos=20.0, neg=-20.0)

        u = finite_tensor(self.fc_u(h), nan=0.0, pos=10.0, neg=-10.0)
        logvar = finite_tensor(self.fc_logvar(h), nan=0.0, pos=6.0, neg=-6.0).clamp(-6.0, 6.0)
        z = finite_tensor(self.reparameterize(u, logvar), nan=0.0, pos=10.0, neg=-10.0)
        recon = finite_tensor(self.decoder(torch.cat([z, coords], dim=1)), nan=0.0, pos=10.0, neg=-10.0)

        return {
            "u": u,
            "mu": u,
            "logvar": logvar,
            "z": z,
            "recon": recon,
            "transformer_feat": transformer_feat,
            "cvae_encoder_input": enc_in,
        }


class PaperSSNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_sp: int,
        grid_res: int,
        ssn_iters: int = 3,
        local_k: int = 9,
        feat_scale: float = 1.0,
        coord_scale: float = 0.35,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.n_sp = n_sp
        self.grid_res = grid_res
        self.ssn_iters = ssn_iters
        self.local_k = local_k
        self.feat_scale = feat_scale
        self.coord_scale = coord_scale

        self.rows, self.cols = self._infer_grid_shape(n_sp)

        self.center_feat_offset = nn.Parameter(torch.zeros(n_sp, in_channels))
        self.center_coord_offset = nn.Parameter(torch.zeros(n_sp, 2))

        nn.init.normal_(self.center_feat_offset, std=0.02)
        nn.init.normal_(self.center_coord_offset, std=0.01)

    @staticmethod
    def _infer_grid_shape(n_sp: int) -> Tuple[int, int]:
        cols = int(round(math.sqrt(n_sp)))
        cols = max(cols, 1)

        rows = int(math.ceil(n_sp / cols))
        rows = max(rows, 1)

        return rows, cols

    def _regular_grid_init(
        self,
        feat: torch.Tensor,
        coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = finite_tensor(feat, nan=0.0, pos=10.0, neg=-10.0)

        coords = finite_tensor(
            coords,
            nan=0.0,
            pos=1.0,
            neg=0.0,
        ).clamp(0.0, 1.0)

        device = feat.device
        n, c = feat.shape

        x = coords[:, 0]
        y = coords[:, 1]

        col_idx = torch.clamp((x * self.cols).long(), 0, self.cols - 1)
        row_idx = torch.clamp((y * self.rows).long(), 0, self.rows - 1)

        cell_idx = row_idx * self.cols + col_idx
        cell_idx = torch.clamp(cell_idx, 0, self.n_sp - 1)

        sp_feat = torch.zeros(self.n_sp, c, device=device)
        sp_coord = torch.zeros(self.n_sp, 2, device=device)
        counts = torch.zeros(self.n_sp, 1, device=device)

        ones = torch.ones(n, 1, device=device)

        sp_feat.index_add_(0, cell_idx, feat)
        sp_coord.index_add_(0, cell_idx, coords)
        counts.index_add_(0, cell_idx, ones)

        valid = counts.squeeze(1) > 0

        sp_feat[valid] = sp_feat[valid] / counts[valid]
        sp_coord[valid] = sp_coord[valid] / counts[valid]

        if (~valid).any():
            sp_feat[~valid] = feat.mean(dim=0, keepdim=True)
            sp_coord[~valid] = coords.mean(dim=0, keepdim=True)

        sp_feat = sp_feat + self.center_feat_offset
        sp_coord = sp_coord + self.center_coord_offset
        sp_coord = sp_coord.clamp(0.0, 1.0)

        return sp_feat, sp_coord

    def _local_candidate_mask(
        self,
        coords: torch.Tensor,
        sp_coord: torch.Tensor,
    ) -> torch.Tensor:
        n = coords.size(0)
        m = sp_coord.size(0)
        k = min(self.local_k, m)

        dist_xy = torch.cdist(coords, sp_coord, p=2)

        nn_idx = torch.topk(
            dist_xy,
            k=k,
            largest=False,
            dim=1,
        ).indices

        mask = torch.zeros(n, m, device=coords.device, dtype=torch.bool)

        row_idx = torch.arange(
            n,
            device=coords.device,
        ).unsqueeze(1).expand(-1, k)

        mask[row_idx, nn_idx] = True

        return mask

    def _assignment(
        self,
        feat: torch.Tensor,
        coords: torch.Tensor,
        sp_feat: torch.Tensor,
        sp_coord: torch.Tensor,
    ) -> torch.Tensor:

        feat = finite_tensor(feat, nan=0.0, pos=10.0, neg=-10.0)
        coords = finite_tensor(coords, nan=0.0, pos=1.0, neg=0.0).clamp(0.0, 1.0)
        sp_feat = finite_tensor(sp_feat, nan=0.0, pos=10.0, neg=-10.0)
        sp_coord = finite_tensor(sp_coord, nan=0.0, pos=1.0, neg=0.0).clamp(0.0, 1.0)

        n = feat.size(0)
        m = sp_feat.size(0)
        chunk_size = int(max(256, globals().get("SSN_ASSIGN_CHUNK_SIZE", 2048)))
        k = min(int(self.local_k), m)

        with torch.no_grad():
            q = torch.empty((n, m), device=feat.device, dtype=feat.dtype)
            feat_scaled = feat.detach() * self.feat_scale
            coord_scaled = coords.detach() * self.coord_scale
            sp_feat_scaled = sp_feat.detach() * self.feat_scale
            sp_coord_scaled = sp_coord.detach() * self.coord_scale

            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                feat_dist = torch.cdist(feat_scaled[start:end], sp_feat_scaled, p=2) ** 2
                coord_dist = torch.cdist(coord_scaled[start:end], sp_coord_scaled, p=2) ** 2
                dist = finite_tensor(feat_dist + coord_dist, nan=1e6, pos=1e6, neg=0.0)

                if k < m:
                    nn_idx = torch.topk(dist, k=k, largest=False, dim=1).indices
                    local_dist = torch.full_like(dist, 1e6)
                    local_dist.scatter_(1, nn_idx, dist.gather(1, nn_idx))
                    dist = local_dist

                q_chunk = F.softmax(-dist, dim=1)
                q_chunk = finite_tensor(q_chunk, nan=0.0, pos=1.0, neg=0.0)
                q_chunk = q_chunk / q_chunk.sum(dim=1, keepdim=True).clamp_min(1e-6)
                q[start:end] = q_chunk

        return q

    def _update_centers(
        self,
        feat: torch.Tensor,
        coords: torch.Tensor,
        q: torch.Tensor,
    ):
        feat = finite_tensor(feat, nan=0.0, pos=10.0, neg=-10.0)

        coords = finite_tensor(
            coords,
            nan=0.0,
            pos=1.0,
            neg=0.0,
        ).clamp(0.0, 1.0)

        q = finite_tensor(q, nan=0.0, pos=1.0, neg=0.0)
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-6)

        mass = q.sum(dim=0, keepdim=True).t().clamp_min(1e-6)

        sp_feat = torch.matmul(q.t(), feat) / mass
        sp_coord = torch.matmul(q.t(), coords) / mass

        sp_feat = finite_tensor(sp_feat, nan=0.0, pos=10.0, neg=-10.0)

        sp_coord = finite_tensor(
            sp_coord,
            nan=0.0,
            pos=1.0,
            neg=0.0,
        ).clamp(0.0, 1.0)

        return sp_feat, sp_coord, mass

    def _build_spatial_graph(
        self,
        sp_feat: torch.Tensor,
        sp_coord: torch.Tensor,
        q: torch.Tensor,
    ):

        sp_feat = finite_tensor(sp_feat, nan=0.0, pos=10.0, neg=-10.0)
        sp_coord = finite_tensor(sp_coord, nan=0.0, pos=1.0, neg=0.0).clamp(0.0, 1.0)
        q = finite_tensor(q, nan=0.0, pos=1.0, neg=0.0)
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-6)

        device = sp_feat.device
        n_sp = sp_feat.size(0)
        labels = torch.argmax(q, dim=1).long().clamp(0, n_sp - 1)

        expected_n = int(self.grid_res * self.grid_res)
        if labels.numel() != expected_n:
            return self._build_fallback_mutual_knn_graph(sp_feat, sp_coord)

        label_map = labels.view(self.grid_res, self.grid_res)

        right_a = label_map[:, :-1].reshape(-1)
        right_b = label_map[:, 1:].reshape(-1)
        down_a = label_map[:-1, :].reshape(-1)
        down_b = label_map[1:, :].reshape(-1)

        src = torch.cat([right_a, down_a], dim=0)
        dst = torch.cat([right_b, down_b], dim=0)
        boundary_mask = src != dst
        src = src[boundary_mask]
        dst = dst[boundary_mask]

        if src.numel() == 0:
            loop = torch.arange(n_sp, device=device)
            edge_index = torch.stack([loop, loop], dim=0).contiguous()
            edge_attr = torch.ones((n_sp, 1), device=device, dtype=torch.float32)
            return edge_index, edge_attr, edge_attr.view(-1)


        a = torch.minimum(src, dst)
        b = torch.maximum(src, dst)
        undirected = torch.unique(torch.stack([a, b], dim=1), dim=0)
        edge_pair = torch.cat([undirected, undirected[:, [1, 0]]], dim=0)

        loop = torch.arange(n_sp, device=device)
        loop_pair = torch.stack([loop, loop], dim=1)
        edge_pair = torch.cat([edge_pair, loop_pair], dim=0)

        edge_src = edge_pair[:, 0].long()
        edge_dst = edge_pair[:, 1].long()

        feat_norm = F.normalize(sp_feat, dim=-1, eps=1e-6)
        cosine = (feat_norm[edge_src] * feat_norm[edge_dst]).sum(dim=1)
        cosine = torch.clamp(cosine, min=0.0, max=1.0)

        dist = torch.norm(sp_coord[edge_src] - sp_coord[edge_dst], p=2, dim=1)
        non_loop = edge_src != edge_dst
        sigma = torch.median(dist[non_loop]).clamp_min(1e-6) if torch.any(non_loop) else torch.tensor(1.0, device=device, dtype=dist.dtype)
        dist_weight = torch.exp(-(dist ** 2) / (2.0 * sigma ** 2))
        edge_weight = torch.where(non_loop, dist_weight * cosine, torch.ones_like(dist))


        keep = (~non_loop) | (edge_weight > 1e-6)
        edge_src = edge_src[keep]
        edge_dst = edge_dst[keep]
        edge_weight = edge_weight[keep]

        edge_index = torch.stack([edge_src, edge_dst], dim=0).contiguous()
        edge_attr = edge_weight.view(-1, 1).contiguous().float()
        return edge_index, edge_attr, edge_weight.float()

    def _build_fallback_mutual_knn_graph(
        self,
        sp_feat: torch.Tensor,
        sp_coord: torch.Tensor,
        k_spatial: int = 8,
    ):

        device = sp_feat.device
        n_sp = sp_feat.size(0)

        if n_sp <= 1:
            loop = torch.arange(n_sp, device=device)
            edge_index = torch.stack([loop, loop], dim=0).contiguous()
            edge_attr = torch.ones((max(n_sp, 1), 1), device=device, dtype=torch.float32)
            return edge_index, edge_attr, edge_attr.view(-1)

        dist_mat = torch.cdist(sp_coord, sp_coord, p=2)
        eye_mask = torch.eye(n_sp, device=device, dtype=torch.bool)
        dist_masked = dist_mat.clone()
        dist_masked[eye_mask] = float("inf")

        k = min(k_spatial, max(1, n_sp - 1))
        nn_idx = torch.topk(dist_masked, k=k, largest=False, dim=1).indices
        knn = torch.zeros((n_sp, n_sp), device=device, dtype=torch.bool)
        rows = torch.arange(n_sp, device=device).unsqueeze(1).expand(-1, k)
        knn[rows, nn_idx] = True

        mutual = knn & knn.t()
        mutual.fill_diagonal_(True)
        edge_pair = mutual.nonzero(as_tuple=False)
        edge_src = edge_pair[:, 0].long()
        edge_dst = edge_pair[:, 1].long()

        feat_norm = F.normalize(sp_feat, dim=-1, eps=1e-6)
        cosine = (feat_norm[edge_src] * feat_norm[edge_dst]).sum(dim=1).clamp(0.0, 1.0)
        dist = torch.norm(sp_coord[edge_src] - sp_coord[edge_dst], p=2, dim=1)
        non_loop = edge_src != edge_dst
        sigma = torch.median(dist[non_loop]).clamp_min(1e-6) if torch.any(non_loop) else torch.tensor(1.0, device=device)
        dist_weight = torch.exp(-(dist ** 2) / (2.0 * sigma ** 2))
        edge_weight = torch.where(non_loop, dist_weight * cosine, torch.ones_like(dist))
        keep = (~non_loop) | (edge_weight > 1e-6)

        edge_src = edge_src[keep]
        edge_dst = edge_dst[keep]
        edge_weight = edge_weight[keep]
        edge_index = torch.stack([edge_src, edge_dst], dim=0).contiguous()
        edge_attr = edge_weight.view(-1, 1).contiguous().float()
        return edge_index, edge_attr, edge_weight.float()

    def forward(self, feat: torch.Tensor, coords: torch.Tensor):
        sp_feat, sp_coord = self._regular_grid_init(feat, coords)

        q = None
        mass = None

        for _ in range(self.ssn_iters):
            q = self._assignment(feat, coords, sp_feat, sp_coord)
            sp_feat, sp_coord, mass = self._update_centers(feat, coords, q)

        edge_index, edge_attr, edge_weight = self._build_spatial_graph(sp_feat, sp_coord, q)

        sp_feature_with_coord = torch.cat([sp_feat, sp_coord], dim=-1)

        aux = {
            "Q": q,
            "sp_feat": sp_feat,
            "sp_coord": sp_coord,
            "mass": mass.squeeze(1),
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_weight": edge_weight,
            "edge_mode": "RAG_distance_cosine_prior_bias_GATv2",
        }

        return sp_feature_with_coord, edge_index, edge_attr, aux


class PriorGATv2Conv(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = False,
        dropout: float = 0.10,
        edge_dim: int = 1,
        beta: float = GAT_PRIOR_BETA,
        eps: float = GAT_EDGE_PRIOR_EPS,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.heads = int(heads)
        self.concat = bool(concat)
        self.dropout = float(dropout)
        self.edge_dim = int(edge_dim)
        self.beta = float(beta)
        self.eps = float(eps)
        self.negative_slope = float(negative_slope)

        self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.empty(heads, out_channels))

        bias_dim = heads * out_channels if concat else out_channels
        self.bias = nn.Parameter(torch.zeros(bias_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        x = finite_tensor(x, nan=0.0, pos=10.0, neg=-10.0)
        n_nodes = x.size(0)

        if edge_index is None or edge_index.numel() == 0:
            out = self.lin_src(x).view(n_nodes, self.heads, self.out_channels)
            out = out.reshape(n_nodes, self.heads * self.out_channels) if self.concat else out.mean(dim=1)
            return finite_tensor(out + self.bias, nan=0.0, pos=10.0, neg=-10.0)

        src = edge_index[0].long()
        dst = edge_index[1].long()
        n_edges = src.numel()

        if edge_attr is None:
            edge_attr = x.new_ones((n_edges, self.edge_dim))
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        edge_attr = edge_attr.to(device=x.device, dtype=x.dtype)
        edge_prior = edge_attr[:, 0].clamp_min(self.eps)

        x_src = self.lin_src(x).view(n_nodes, self.heads, self.out_channels)
        x_dst = self.lin_dst(x).view(n_nodes, self.heads, self.out_channels)
        e_emb = self.lin_edge(edge_attr).view(n_edges, self.heads, self.out_channels)


        att_input = x_dst[dst] + x_src[src] + e_emb
        att_input = F.leaky_relu(att_input, negative_slope=self.negative_slope)
        score = (att_input * self.att.unsqueeze(0)).sum(dim=-1)


        score = score + self.beta * torch.log(edge_prior).unsqueeze(-1)
        score = finite_tensor(score, nan=-1e4, pos=1e4, neg=-1e4)

        alpha = pyg_softmax(score, dst, num_nodes=n_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = x_src[src] * alpha.unsqueeze(-1)
        out = x.new_zeros((n_nodes, self.heads, self.out_channels))
        out.index_add_(0, dst, msg)

        if self.concat:
            out = out.reshape(n_nodes, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        out = out + self.bias
        return finite_tensor(out, nan=0.0, pos=10.0, neg=-10.0)


class PaperSSNGATBranch(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_sp: int,
        grid_res: int,
        ssn_iters: int = 3,
        local_k: int = 9,
    ):
        super().__init__()

        self.ssn = PaperSSNLayer(
            in_dim,
            n_sp,
            grid_res,
            ssn_iters,
            local_k,
            feat_scale=1.0,
            coord_scale=0.35,
        )

        self.gat = PriorGATv2Conv(
            in_channels=in_dim + 2,
            out_channels=out_dim,
            heads=4,
            concat=False,
            dropout=0.10,
            edge_dim=1,
            beta=GAT_PRIOR_BETA,
            eps=GAT_EDGE_PRIOR_EPS,
        )

    def forward(
        self,
        feat: torch.Tensor,
        coords: torch.Tensor,
        return_aux: bool = False,
    ):
        sp_feat, edge_index, edge_attr, aux = self.ssn(feat, coords)

        h_sp = self.gat(sp_feat, edge_index, edge_attr=edge_attr)
        h_sp = finite_tensor(F.elu(h_sp), nan=0.0, pos=10.0, neg=-10.0)

        pixel_feat = torch.matmul(aux["Q"], h_sp)
        pixel_feat = finite_tensor(pixel_feat, nan=0.0, pos=10.0, neg=-10.0)

        if return_aux:
            aux["pixel_feat"] = pixel_feat
            return pixel_feat, aux

        return pixel_feat


class MultiScaleAttentionFusion(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()

        self.score_net = nn.Sequential(
            nn.Linear(in_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
    ):
        f1 = finite_tensor(f1, nan=0.0, pos=10.0, neg=-10.0)
        f2 = finite_tensor(f2, nan=0.0, pos=10.0, neg=-10.0)
        f3 = finite_tensor(f3, nan=0.0, pos=10.0, neg=-10.0)

        g = torch.cat(
            [
                f1.mean(dim=0),
                f2.mean(dim=0),
                f3.mean(dim=0),
            ],
            dim=0,
        )

        weights = F.softmax(
            self.score_net(
                finite_tensor(g, nan=0.0, pos=10.0, neg=-10.0)
            ),
            dim=0,
        )

        weights = finite_tensor(weights, nan=1.0 / 3.0, pos=1.0, neg=0.0)
        weights = weights / weights.sum().clamp_min(1e-6)

        fused = weights[0] * f1 + weights[1] * f2 + weights[2] * f3
        fused = finite_tensor(fused, nan=0.0, pos=10.0, neg=-10.0)

        return fused, weights


class FrontBackAttentionFusion(nn.Module):
    def __init__(
        self,
        front_dim: int,
        back_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 128,
    ):
        super().__init__()

        self.front_proj = nn.Linear(front_dim, out_dim)
        self.back_proj = nn.Linear(back_dim, out_dim)

        self.branch_attn = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        mid_dim = max(out_dim // 2, 16)

        self.channel_attn = nn.Sequential(
            nn.Linear(out_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, out_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        f_front: torch.Tensor,
        f_back: torch.Tensor,
    ):
        f_front = finite_tensor(f_front, nan=0.0, pos=10.0, neg=-10.0)
        f_back = finite_tensor(f_back, nan=0.0, pos=10.0, neg=-10.0)

        f_front = finite_tensor(
            self.front_proj(f_front),
            nan=0.0,
            pos=10.0,
            neg=-10.0,
        )

        f_back = finite_tensor(
            self.back_proj(f_back),
            nan=0.0,
            pos=10.0,
            neg=-10.0,
        )

        attn = torch.softmax(
            self.branch_attn(torch.cat([f_front, f_back], dim=1)),
            dim=1,
        )

        attn = finite_tensor(attn, nan=0.5, pos=1.0, neg=0.0)
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)

        fused = attn[:, 0:1] * f_front + attn[:, 1:2] * f_back

        ch_weight = self.channel_attn(fused)
        ch_weight = finite_tensor(ch_weight, nan=1.0, pos=1.0, neg=0.0)

        out = fused * ch_weight
        out = finite_tensor(out, nan=0.0, pos=10.0, neg=-10.0)

        return out, attn, ch_weight

class DualBranchAttentionModel(nn.Module):

    def __init__(
        self,
        feat_dim: int = 7,
        grid_res: int = 200,
        cnn_dim: int = 128,
        vae_hidden: int = 128,
        vae_latent: int = PROJ_DIM,
        dropout_rate: float = DROPOUT,
        model_variant: str = None,
    ):
        super().__init__()
        del cnn_dim

        self.model_variant = str(model_variant or globals().get("MODEL_VARIANT", "transformer_cvae_ssn_gat_backend_u"))
        allowed = {
            "transformer_cvae_ssn_gat_backend_u",
            "transformer_cvae_ssn_gat_backend_proj_u",
        }
        if self.model_variant not in allowed:
            raise ValueError(f"MODEL_VARIANT 必须为 {sorted(allowed)}，当前={self.model_variant}")

        self.cvae = CoordCVAE(
            feat_dim=feat_dim,
            d_model=TRANSFORMER_D_MODEL,
            coord_dim=2,
            hidden=vae_hidden,
            latent=vae_latent,
            dropout=0.10,
        )
        self.u_norm = nn.LayerNorm(vae_latent)
        self.projection_head = nn.Sequential(
            nn.Linear(vae_latent, PROJECTION_HEAD_HIDDEN_DIM),
            nn.LayerNorm(PROJECTION_HEAD_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(PROJECTION_HEAD_HIDDEN_DIM, PROJECTION_HEAD_DIM),
        )

        self.backend_input_name = "u" if self.model_variant.endswith("backend_u") else "proj(u)"
        ssn_in_dim = int(vae_latent) if self.backend_input_name == "u" else int(PROJECTION_HEAD_DIM)

        self.ssn_scales = list(SSN_SCALES)
        if len(self.ssn_scales) != 3:
            raise ValueError(f"SSN_SCALES 必须正好包含 3 个尺度，当前={self.ssn_scales}")
        self.ssn_branches = nn.ModuleList([
            PaperSSNGATBranch(ssn_in_dim, 64, int(n_sp), grid_res)
            for n_sp in self.ssn_scales
        ])
        self.ms_fusion = MultiScaleAttentionFusion(64)

        self.fb_fusion = FrontBackAttentionFusion(
            front_dim=vae_latent + PROJECTION_HEAD_DIM,
            back_dim=64,
            hidden_dim=128,
            out_dim=128,
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x_feat: torch.Tensor,
        coords: torch.Tensor,
        inject_noise: bool = False,
        return_aux: bool = False,
    ):
        x_feat = finite_tensor(x_feat, nan=0.0, pos=8.0, neg=-8.0).float()
        coords = finite_tensor(coords, nan=0.0, pos=1.0, neg=0.0).float().clamp(0.0, 1.0)


        cvae_out = self.cvae(x_feat, coords, inject_noise=inject_noise)
        transformer_feat = cvae_out["transformer_feat"]


        u_raw = finite_tensor(cvae_out["u"], nan=0.0, pos=10.0, neg=-10.0)
        if inject_noise and self.training:
            u_raw = u_raw + torch.randn_like(u_raw) * 0.01
        u_feat = finite_tensor(self.u_norm(u_raw), nan=0.0, pos=10.0, neg=-10.0)

        proj_emb = finite_tensor(
            self.projection_head(u_feat),
            nan=0.0,
            pos=10.0,
            neg=-10.0,
        )
        proj_emb = F.normalize(proj_emb, dim=1, eps=1e-6)
        front_fusion_input = finite_tensor(torch.cat([u_feat, proj_emb], dim=1), nan=0.0, pos=10.0, neg=-10.0)


        if self.backend_input_name == "u":
            backend_input = u_feat
        else:
            backend_input = proj_emb
        backend_input = finite_tensor(backend_input, nan=0.0, pos=10.0, neg=-10.0)

        multi_feats = [branch(backend_input, coords, return_aux=False) for branch in self.ssn_branches]

        if inject_noise and self.training:
            multi_feats = [f + torch.randn_like(f) * 0.02 for f in multi_feats]

        back_feat, ms_weights = self.ms_fusion(multi_feats[0], multi_feats[1], multi_feats[2])
        fusion_feat, branch_attn, channel_attn = self.fb_fusion(front_fusion_input, back_feat)
        logits = self.classifier(fusion_feat)
        logits = finite_tensor(logits, nan=0.0, pos=20.0, neg=-20.0)

        if not return_aux:
            return logits

        return logits, {
            "model_variant": self.model_variant,
            "backend_input_name": self.backend_input_name,
            "ssn_scales": torch.as_tensor(self.ssn_scales, device=logits.device),
            "ms_weights": ms_weights,
            "branch_attn": branch_attn,
            "channel_attn": channel_attn,
            "transformer_feat": transformer_feat,
            "front_feat": u_feat,
            "front_fusion_input": front_fusion_input,
            "back_feat": back_feat,
            "backend_input": backend_input,
            "proj_emb": proj_emb,
            "proj_u": proj_emb,
            "u": u_feat,
            "mu": cvae_out["mu"],
            "logvar": cvae_out["logvar"],
            "z": cvae_out["z"],
            "recon": cvae_out["recon"],
        }





def weighted_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor = None,
    alpha: float = FOCAL_ALPHA,
    gamma: float = FOCAL_GAMMA,
):

    logits = finite_tensor(logits, nan=0.0, pos=20.0, neg=-20.0)

    targets = finite_tensor(
        targets,
        nan=0.0,
        pos=1.0,
        neg=0.0,
    ).float()
    targets = torch.clamp(targets, 0.0, 1.0)

    if logits.shape != targets.shape:
        targets = targets.view_as(logits)

    bce_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    p_t = torch.clamp(p_t, min=1e-6, max=1.0 - 1e-6)

    alpha_t = float(alpha) * targets + (1.0 - float(alpha)) * (1.0 - targets)
    focal_factor = torch.pow(1.0 - p_t, float(gamma))

    if weights is not None:
        weights = finite_tensor(weights, nan=1.0, pos=10.0, neg=0.0).float()

        if weights.shape != bce_loss.shape:
            weights = weights.view_as(bce_loss)

        effective_weights = weights * alpha_t
        weighted_loss = focal_factor * bce_loss * effective_weights
        denom = effective_weights.sum().clamp_min(1e-6)

        return finite_scalar_loss(weighted_loss.sum() / denom, device=logits.device)

    loss = alpha_t * focal_factor * bce_loss
    return finite_scalar_loss(loss.mean(), device=logits.device)




def spatial_smoothness_loss(
    logits: torch.Tensor,
    grid_res: int,
    work_area_mask_np: np.ndarray = None,
) -> torch.Tensor:

    logits = finite_tensor(logits, nan=0.0, pos=20.0, neg=-20.0)
    probs = torch.sigmoid(logits).view(grid_res, grid_res)

    dx = torch.abs(probs[:, 1:] - probs[:, :-1])
    dy = torch.abs(probs[1:, :] - probs[:-1, :])

    if work_area_mask_np is not None:
        mask_np = np.asarray(work_area_mask_np, dtype=bool)
        if mask_np.shape == (grid_res, grid_res) and np.any(mask_np):
            mask = torch.as_tensor(mask_np, dtype=torch.bool, device=logits.device)
            mask_x = mask[:, 1:] & mask[:, :-1]
            mask_y = mask[1:, :] & mask[:-1, :]

            parts = []
            if torch.any(mask_x):
                parts.append(dx[mask_x].mean())
            if torch.any(mask_y):
                parts.append(dy[mask_y].mean())

            if len(parts) > 0:
                return finite_scalar_loss(torch.stack(parts).mean(), device=logits.device)

    return finite_scalar_loss(0.5 * (dx.mean() + dy.mean()), device=logits.device)


def reconstruction_loss(recon: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:

    if recon is None or target_feat is None:
        device = recon.device if torch.is_tensor(recon) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros((), device=device, dtype=torch.float32)
    recon = finite_tensor(recon, nan=0.0, pos=10.0, neg=-10.0).float()
    target_feat = finite_tensor(target_feat.detach(), nan=0.0, pos=10.0, neg=-10.0).float()
    if recon.shape != target_feat.shape:
        return torch.zeros((), device=recon.device, dtype=torch.float32)
    return finite_scalar_loss(F.mse_loss(recon, target_feat, reduction="mean"), device=recon.device)


def kl_divergence_loss(u: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:

    if u is None or logvar is None:
        device = u.device if torch.is_tensor(u) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros((), device=device, dtype=torch.float32)
    u = finite_tensor(u, nan=0.0, pos=10.0, neg=-10.0).float()
    logvar = finite_tensor(logvar, nan=0.0, pos=6.0, neg=-6.0).float().clamp(-6.0, 6.0)
    kl = -0.5 * (1.0 + logvar - u.pow(2) - logvar.exp())
    return finite_scalar_loss(kl.mean(), device=u.device)


def _flatten_grads(grads, params: list) -> torch.Tensor:
    flat = []
    for grad, param in zip(grads, params):
        if grad is None:
            flat.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))
        else:
            flat.append(finite_tensor(grad, nan=0.0, pos=1e4, neg=-1e4).reshape(-1))
    return torch.cat(flat) if flat else torch.zeros(0)


def _assign_flat_grad(params: list, flat_grad: torch.Tensor):
    offset = 0
    for param in params:
        n = param.numel()
        param.grad = flat_grad[offset: offset + n].view_as(param).detach().clone()
        offset += n


def cagrad_backward(
    loss_dict: Dict[str, torch.Tensor],
    model: nn.Module,
    alpha: float = CAGRAD_ALPHA,
    rescale: int = CAGRAD_RESCALE,
    retain_graph_for_later_backward: bool = False,
) -> Dict:
    first_param = next(model.parameters())
    active = []
    for name, loss_value in loss_dict.items():
        if loss_value is None or not torch.is_tensor(loss_value):
            continue
        loss_value = finite_scalar_loss(loss_value, device=first_param.device)
        if not loss_value.requires_grad:
            continue
        if abs(float(loss_value.detach().cpu())) <= 1e-12:
            continue
        active.append((name, loss_value))

    if len(active) == 0:
        return {"cagrad_active": 0, "cagrad_weights": ""}
    if len(active) == 1:
        active[0][1].backward(retain_graph=retain_graph_for_later_backward)
        return {"cagrad_active": 1, "cagrad_weights": active[0][0]}

    params = [p for p in model.parameters() if p.requires_grad]
    grads = []
    for i, (_, loss_value) in enumerate(active):
        grad_tuple = torch.autograd.grad(
            loss_value,
            params,
            retain_graph=retain_graph_for_later_backward or (i < len(active) - 1),
            allow_unused=True,
        )
        grads.append(_flatten_grads(grad_tuple, params))

    grad_mat = finite_tensor(torch.stack(grads, dim=0), nan=0.0, pos=1e4, neg=-1e4)
    g0 = grad_mat.mean(dim=0)
    g0_norm = g0.norm().clamp_min(1e-12)
    gram = grad_mat.matmul(grad_mat.t()).detach().cpu().double().numpy()
    n_tasks = len(active)
    x_start = np.ones(n_tasks, dtype=np.float64) / max(n_tasks, 1)
    bounds = tuple((0.0, 1.0) for _ in range(n_tasks))
    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1.0},)
    c = float(alpha) * float(g0_norm.detach().cpu())

    def objfn(x):
        x = np.asarray(x, dtype=np.float64)
        gw_norm = float(np.sqrt(max(x.dot(gram).dot(x), 0.0) + 1e-12))
        linear = float(x.dot(gram).dot(x_start))
        return linear + c * gw_norm

    try:
        res = minimize(objfn, x_start, bounds=bounds, constraints=constraints, method="SLSQP")
        weights = res.x if res.success and np.all(np.isfinite(res.x)) else x_start
    except Exception:
        weights = x_start

    weights_t = torch.as_tensor(weights, dtype=grad_mat.dtype, device=grad_mat.device)
    gw = torch.sum(grad_mat * weights_t.view(-1, 1), dim=0)
    gw_norm = gw.norm().clamp_min(1e-12)
    merged = g0 + (c / float(gw_norm.detach().cpu())) * gw
    if int(rescale) == 1:
        merged = merged / (1.0 + float(alpha) ** 2)
    elif int(rescale) == 2:
        merged = merged / (1.0 + float(alpha))
    _assign_flat_grad(params, merged)
    weight_txt = ",".join(f"{name}:{float(w):.3f}" for (name, _), w in zip(active, weights))
    return {"cagrad_active": n_tasks, "cagrad_weights": weight_txt}


def consistency_loss_with_ema_teacher(
    logits: torch.Tensor,
    ema_probs_np: np.ndarray,
    candidate_indices_np: np.ndarray,
    conf_margin: float = 0.10,
    max_samples: int = 12000,
) -> torch.Tensor:

    if ema_probs_np is None or candidate_indices_np is None:
        return torch.zeros((), device=logits.device, dtype=torch.float32)

    candidate_indices_np = np.asarray(candidate_indices_np, dtype=np.int64).reshape(-1)
    if candidate_indices_np.size == 0:
        return torch.zeros((), device=logits.device, dtype=torch.float32)

    flat_probs = torch.sigmoid(finite_tensor(logits, nan=0.0, pos=20.0, neg=-20.0)).view(-1)

    valid_np = candidate_indices_np[
        (candidate_indices_np >= 0) & (candidate_indices_np < flat_probs.numel())
    ]
    if valid_np.size == 0:
        return torch.zeros((), device=logits.device, dtype=torch.float32)

    if valid_np.size > int(max_samples):
        choice = np.random.choice(valid_np.size, size=int(max_samples), replace=False)
        valid_np = valid_np[choice]

    idx_t = torch.as_tensor(valid_np, dtype=torch.long, device=logits.device)
    teacher_t = torch.as_tensor(
        np.asarray(ema_probs_np, dtype=np.float32)[valid_np],
        dtype=torch.float32,
        device=logits.device,
    ).clamp(0.0, 1.0).detach()

    conf_mask = torch.abs(teacher_t - 0.5) >= float(conf_margin)
    if not torch.any(conf_mask):
        return torch.zeros((), device=logits.device, dtype=torch.float32)

    student = flat_probs[idx_t][conf_mask]
    teacher = teacher_t[conf_mask]

    return finite_scalar_loss(F.mse_loss(student, teacher, reduction="mean"), device=logits.device)





@torch.no_grad()
def mc_teacher_predict(
    model: nn.Module,
    x_data: torch.Tensor,
    coords: torch.Tensor,
    mc_times: int = 1,
):

    del mc_times

    was_training = model.training
    model.eval()

    logits = model(
        x_data,
        coords,
        inject_noise=False,
        return_aux=False,
    )

    mean_probs = torch.sigmoid(
        finite_tensor(logits, nan=0.0, pos=20.0, neg=-20.0)
    ).detach().reshape(-1).cpu().numpy().astype(np.float32)

    std_probs = np.zeros_like(mean_probs, dtype=np.float32)

    if was_training:
        model.train()

    return mean_probs, std_probs

def extract_u_embeddings(
    model: nn.Module,
    x_data: torch.Tensor,
    coords: torch.Tensor,
) -> np.ndarray:

    was_training = model.training
    model.eval()
    with torch.no_grad():
        _, aux = model(x_data, coords, inject_noise=False, return_aux=True)
        u = aux.get("proj_emb", None)
        if u is None:
            u = aux.get("front_feat", None)
        if u is None:
            u = aux.get("transformer_feat", None)
        if u is None:
            u = aux.get("u", aux.get("mu"))
        u = finite_tensor(u, nan=0.0, pos=10.0, neg=-10.0)
        u = F.normalize(u, dim=1, eps=1e-6)
        arr = u.detach().cpu().numpy().astype(np.float32)
    if was_training:
        model.train()
    return arr


def compute_u_projection_similarity(
    proj_emb: np.ndarray,
    candidate_indices: np.ndarray,
    real_oil_indices: np.ndarray,
    real_dry_indices: np.ndarray,
):

    candidate_indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    n = candidate_indices.size
    oil_sim = np.full(n, -1.0, dtype=np.float64)
    dry_sim = np.full(n, -1.0, dtype=np.float64)

    if proj_emb is None or np.asarray(proj_emb).ndim != 2 or n == 0:
        return oil_sim, dry_sim

    emb = np.asarray(proj_emb, dtype=np.float64)
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
    n_total = emb.shape[0]
    valid = (candidate_indices >= 0) & (candidate_indices < n_total)

    real_oil_indices = np.asarray(real_oil_indices if real_oil_indices is not None else [], dtype=np.int64).reshape(-1)
    real_dry_indices = np.asarray(real_dry_indices if real_dry_indices is not None else [], dtype=np.int64).reshape(-1)
    real_oil_indices = real_oil_indices[(real_oil_indices >= 0) & (real_oil_indices < n_total)]
    real_dry_indices = real_dry_indices[(real_dry_indices >= 0) & (real_dry_indices < n_total)]

    if not np.any(valid):
        return oil_sim, dry_sim

    cand = emb[candidate_indices[valid]]
    if real_oil_indices.size > 0:
        oil_center = emb[real_oil_indices].mean(axis=0, keepdims=True)
        oil_center = oil_center / np.clip(np.linalg.norm(oil_center, axis=1, keepdims=True), 1e-8, None)
        oil_sim[valid] = cand.dot(oil_center.T).reshape(-1)
    if real_dry_indices.size > 0:
        dry_center = emb[real_dry_indices].mean(axis=0, keepdims=True)
        dry_center = dry_center / np.clip(np.linalg.norm(dry_center, axis=1, keepdims=True), 1e-8, None)
        dry_sim[valid] = cand.dot(dry_center.T).reshape(-1)
    return oil_sim, dry_sim


def schedule_stage2_topk(
    epoch: int,
    total_epochs: int,
    warmup: int,
    cfg: Dict,
    prefix: str,
) -> int:
    extra = cfg.get("balanced_extra_per_class", None)
    if extra is not None:
        return int(max(0, extra))

    start = int(cfg.get(f"stage2_{prefix}_topk_start", cfg.get("stage2_topk_start", -1)))
    mid = int(cfg.get(f"stage2_{prefix}_topk_mid", cfg.get("stage2_topk_mid", start)))
    end = int(cfg.get(f"stage2_{prefix}_topk_end", cfg.get("stage2_topk_end", mid)))
    if start < 0 and mid < 0 and end < 0:
        return -1
    start = max(start, 0)
    mid = max(mid, start)
    end = max(end, mid)
    p = np.clip((epoch - warmup) / max(total_epochs - warmup - 1, 1), 0.0, 1.0)
    if p <= 0.45:
        value = start + (mid - start) * (p / 0.45)
    else:
        value = mid + (end - mid) * ((p - 0.45) / 0.55)
    return int(max(0, round(value)))



def schedule_stage2_total_topk(
    epoch: int,
    total_epochs: int,
    warmup: int,
    cfg: Dict,
) -> int:

    start = int(cfg.get("stage2_total_topk_start", cfg.get("stage2_topk_start", 300)))
    mid = int(cfg.get("stage2_total_topk_mid", cfg.get("stage2_topk_mid", start)))
    end = int(cfg.get("stage2_total_topk_end", cfg.get("stage2_topk_end", mid)))
    start = max(start, 0)
    mid = max(mid, start)
    end = max(end, mid)
    p = np.clip((epoch - warmup) / max(total_epochs - warmup - 1, 1), 0.0, 1.0)
    if p <= 0.45:
        value = start + (mid - start) * (p / 0.45)
    else:
        value = mid + (end - mid) * ((p - 0.45) / 0.55)
    return int(max(0, round(value)))


def _ema_update_in_cfg(cfg: Dict, key: str, value: float, momentum: float, default: float) -> float:

    value = float(value) if np.isfinite(value) else float(default)
    old = cfg.get(key, None)
    if old is None or not np.isfinite(float(old)):
        new_value = value
    else:
        new_value = float(momentum) * float(old) + (1.0 - float(momentum)) * value
    cfg[key] = float(new_value)
    return float(new_value)


def update_pseudo_labels(
    ema_probs: np.ndarray,
    std_probs: np.ndarray,
    unlabeled_pool: np.ndarray,
    base_trn_labels: np.ndarray,
    pseudo_oil_dict: Dict[int, float],
    pseudo_dry_set: set,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    cfg: Dict,
    proj_emb: np.ndarray = None,
    real_oil_indices: np.ndarray = None,
    real_dry_indices: np.ndarray = None,
):

    del pseudo_oil_dict, pseudo_dry_set, proj_emb, real_oil_indices, real_dry_indices

    real_oil = int(np.sum(base_trn_labels == 1))
    real_dry = int(np.sum(base_trn_labels == 0))
    real_total = max(real_oil + real_dry, 1)
    real_oil_prior = float(real_oil / real_total)
    pseudo_start_epoch = max(int(warmup_epochs), int(cfg.get("stage2_start_epoch", warmup_epochs)))

    unlabeled_pool = np.asarray(unlabeled_pool, dtype=np.int64).reshape(-1)
    if ema_probs is None:
        ema_probs = np.full(
            max(int(unlabeled_pool.max()) + 1 if unlabeled_pool.size else 1, 1),
            0.5,
            dtype=np.float32,
        )
    ema_probs = np.asarray(ema_probs, dtype=np.float32).reshape(-1)

    if std_probs is None:
        std_probs = np.zeros_like(ema_probs, dtype=np.float32)
    std_probs = np.asarray(std_probs, dtype=np.float32).reshape(-1)

    valid_pool = unlabeled_pool[(unlabeled_pool >= 0) & (unlabeled_pool < ema_probs.size)]
    pool_probs = (
        np.nan_to_num(ema_probs[valid_pool], nan=0.5, posinf=1.0, neginf=0.0)
        if valid_pool.size
        else np.array([], dtype=np.float32)
    )
    pool_std = (
        np.nan_to_num(std_probs[valid_pool], nan=0.0, posinf=1.0, neginf=0.0)
        if valid_pool.size and std_probs.size > int(valid_pool.max(initial=0))
        else np.zeros_like(pool_probs, dtype=np.float32)
    )

    conf_oil = np.clip(pool_probs, 0.0, 1.0)
    conf_dry = np.clip(1.0 - pool_probs, 0.0, 1.0)
    pred_is_oil = conf_oil >= conf_dry
    conf_max = np.maximum(conf_oil, conf_dry)

    max_prob_dbg = float(np.nanmax(pool_probs)) if pool_probs.size else float("nan")
    p95_prob_dbg = float(np.nanpercentile(pool_probs, 95)) if pool_probs.size else float("nan")
    mean_std_dbg = float(np.nanmean(pool_std)) if pool_std.size else 0.0

    total_cap = schedule_stage2_total_topk(epoch, total_epochs, pseudo_start_epoch, cfg)
    total_cap = int(min(max(total_cap, 0), max(int(valid_pool.size), 0)))



    if pool_probs.size > 0:
        cur_global_conf = float(np.mean(conf_max))
        cur_oil_conf = float(np.mean(conf_oil[pred_is_oil])) if np.any(pred_is_oil) else float(np.mean(conf_oil))
        cur_dry_conf = float(np.mean(conf_dry[~pred_is_oil])) if np.any(~pred_is_oil) else float(np.mean(conf_dry))
        cur_pred_oil_ratio = float(np.mean(pred_is_oil.astype(np.float32)))
    else:
        cur_global_conf = 0.5
        cur_oil_conf = 0.5
        cur_dry_conf = 0.5
        cur_pred_oil_ratio = real_oil_prior


    ema_global_conf = cur_global_conf
    ema_oil_conf = cur_oil_conf
    ema_dry_conf = cur_dry_conf
    ema_pred_oil_ratio = cur_pred_oil_ratio
    ema_std = mean_std_dbg

    min_global = float(cfg.get("ema_min_global_conf", 0.55))
    max_global = float(cfg.get("ema_max_global_conf", 0.95))
    min_class_conf = float(cfg.get("ema_min_class_conf", 0.55))
    global_thr = float(np.clip(cur_global_conf, min_global, max_global))


    class_conf_max = max(float(cur_oil_conf), float(cur_dry_conf), 1e-6)
    oil_conf_thr = float(np.clip(global_thr * (float(cur_oil_conf) / class_conf_max), min_class_conf, max_global))
    dry_conf_thr = float(np.clip(global_thr * (float(cur_dry_conf) / class_conf_max), min_class_conf, max_global))

    oil_prob_th = oil_conf_thr
    dry_prob_th = 1.0 - dry_conf_thr

    std_gate = float(np.clip(
        max(float(cfg.get("ema_std_min", 0.02)), float(cfg.get("ema_std_multiplier", 2.0)) * ema_std),
        float(cfg.get("ema_std_min", 0.02)),
        float(cfg.get("ema_std_max", 0.20)),
    ))

    empty_debug = {
        "stage": "stage-1-warm-up-real-only",
        "dyn_oil_th": float("nan"),
        "dyn_dry_th": float("nan"),
        "raw_oil": 0,
        "raw_dry": 0,
        "topk_oil_current": 0,
        "topk_dry_current": 0,
        "pseudo_oil": 0,
        "pseudo_dry": 0,
        "eff_pos_neg_ratio": float(real_oil / max(real_dry, 1)),
        "max_unlab_prob": max_prob_dbg,
        "p95_unlab_prob": p95_prob_dbg,
        "mean_unlab_std": mean_std_dbg,
        "oil_relaxed": 0,
        "dry_forced": 0,
        "proj_oil_pass": 0,
        "proj_sim_mean": float("nan"),
        "proj_sim_p95": float("nan"),
        "proj_margin_mean": float("nan"),
        "oil_allow": 0,
        "dry_allow": 0,
        "oil_allow_cap": 0,
        "oil_allow_ratio": 0,
        "dry_allow_cap": 0,
        "dry_allow_ratio": 0,
        "pruned_oil": 0,
        "pruned_dry": 0,
        "adaptive_conf_threshold": float(global_thr),
        "softmatch_weight_mean": float("nan"),
        "csl_score_quantile": float("nan"),
        "csl_oil_score_threshold": float(oil_prob_th),
        "csl_dry_score_threshold": float(dry_prob_th),
        "class_ratio_guard_low": float(cfg.get("ema_min_class_ratio", 0.10)),
        "class_ratio_guard_hi": float(cfg.get("ema_max_class_ratio", 0.90)),
        "pre_guard_oil": 0,
        "pre_guard_dry": 0,
        "scheduled_oil_cap": 0,
        "scheduled_dry_cap": 0,
        "balance_pseudo_pool_capacity": int(True),
        "force_final_pos_neg_1to1": int(True),
        "ema_global_conf": float(ema_global_conf),
        "ema_oil_conf": float(ema_oil_conf),
        "ema_dry_conf": float(ema_dry_conf),
        "ema_pred_oil_ratio": float(ema_pred_oil_ratio),
        "ema_std": float(ema_std),
        "std_gate": float(std_gate),
        "target_oil_ratio": float("nan"),
        "pseudo_quality_mean": float("nan"),
    }

    if epoch < pseudo_start_epoch or valid_pool.size < 2 or total_cap <= 0:
        return {}, {}, empty_debug


    eps = 1e-6
    oil_quality = (conf_oil - oil_prob_th) / max(1.0 - oil_prob_th, eps)
    dry_quality = (conf_dry - dry_conf_thr) / max(1.0 - dry_conf_thr, eps)
    oil_quality = np.clip(oil_quality, 0.0, 1.0)
    dry_quality = np.clip(dry_quality, 0.0, 1.0)

    if bool(cfg.get("use_uncertainty_gate", True)):
        uncertainty_weight = np.clip(1.0 - (pool_std / max(std_gate, eps)), 0.0, 1.0)
        oil_quality = oil_quality * uncertainty_weight
        dry_quality = dry_quality * uncertainty_weight

    oil_candidate_pos = np.where(oil_quality > 0.0)[0]
    dry_candidate_pos = np.where(dry_quality > 0.0)[0]
    raw_oil_count = int(oil_candidate_pos.size)
    raw_dry_count = int(dry_candidate_pos.size)

    if raw_oil_count == 0 and raw_dry_count == 0:
        debug = dict(empty_debug)
        debug.update(
            {
                "stage": "stage-2-qq-ema-no-quality-candidate",
                "dyn_oil_th": float(oil_prob_th),
                "dyn_dry_th": float(dry_prob_th),
                "raw_oil": 0,
                "raw_dry": 0,
                "oil_allow_cap": 0,
                "dry_allow_cap": 0,
                "scheduled_oil_cap": 0,
                "scheduled_dry_cap": 0,
            }
        )
        return {}, {}, debug

    oil_order = (
        oil_candidate_pos[np.argsort(-oil_quality[oil_candidate_pos], kind="mergesort")]
        if oil_candidate_pos.size > 0
        else np.array([], dtype=np.int64)
    )
    dry_order = (
        dry_candidate_pos[np.argsort(-dry_quality[dry_candidate_pos], kind="mergesort")]
        if dry_candidate_pos.size > 0
        else np.array([], dtype=np.int64)
    )






    oil_prefix_quality = np.concatenate([
        np.array([0.0], dtype=np.float64),
        np.cumsum(oil_quality[oil_order].astype(np.float64)) if oil_order.size > 0 else np.array([], dtype=np.float64),
    ])
    dry_prefix_quality = np.concatenate([
        np.array([0.0], dtype=np.float64),
        np.cumsum(dry_quality[dry_order].astype(np.float64)) if dry_order.size > 0 else np.array([], dtype=np.float64),
    ])

    max_oil_keep = int(min(oil_order.size, total_cap))
    max_dry_keep = int(min(dry_order.size, total_cap))

    best_score = None
    keep_oil = 0
    keep_dry = 0

    for ko in range(max_oil_keep + 1):
        kd_upper = int(min(max_dry_keep, total_cap - ko))

        ideal_kd = int(round((real_oil + ko) - real_dry))
        candidate_kd = {
            0,
            kd_upper,
            int(np.clip(ideal_kd, 0, kd_upper)),
            int(np.clip(ideal_kd - 1, 0, kd_upper)),
            int(np.clip(ideal_kd + 1, 0, kd_upper)),
        }
        for kd in candidate_kd:
            final_pos = int(real_oil + ko)
            final_neg = int(real_dry + kd)
            imbalance = abs(final_pos - final_neg)
            selected_total = int(ko + kd)
            quality_sum = float(oil_prefix_quality[ko] + dry_prefix_quality[kd])
            score = (imbalance, -selected_total, -quality_sum)
            if best_score is None or score < best_score:
                best_score = score
                keep_oil = int(ko)
                keep_dry = int(kd)

    oil_cap = int(keep_oil)
    dry_cap = int(keep_dry)
    target_oil_ratio = float(keep_oil / max(keep_oil + keep_dry, 1))

    oil_idx = valid_pool[oil_order[:keep_oil]].astype(np.int64) if keep_oil > 0 else np.array([], dtype=np.int64)
    dry_idx = valid_pool[dry_order[:keep_dry]].astype(np.int64) if keep_dry > 0 else np.array([], dtype=np.int64)

    oil_set = set(int(i) for i in oil_idx.tolist())
    dry_keep_pairs = [
        (int(idx), float(dry_quality[pos]))
        for idx, pos in zip(dry_idx.tolist(), dry_order[:keep_dry].tolist())
        if int(idx) not in oil_set
    ]
    oil_keep_pairs = [(int(idx), float(oil_quality[pos])) for idx, pos in zip(oil_idx.tolist(), oil_order[:keep_oil].tolist())]

    q_min = float(cfg.get("pseudo_quality_weight_min", 0.25))
    q_power = float(cfg.get("pseudo_quality_power", 1.0))
    next_oil = {
        int(idx): float(q_min + (1.0 - q_min) * (np.clip(q, 0.0, 1.0) ** q_power))
        for idx, q in oil_keep_pairs
    }
    next_dry = {
        int(idx): float(q_min + (1.0 - q_min) * (np.clip(q, 0.0, 1.0) ** q_power))
        for idx, q in dry_keep_pairs
        if int(idx) not in next_oil
    }

    qualities = list(next_oil.values()) + list(next_dry.values())
    q_mean = float(np.mean(qualities)) if len(qualities) > 0 else float("nan")
    eff_pos = real_oil + len(next_oil)
    eff_neg = real_dry + len(next_dry)

    return next_oil, next_dry, {
        "stage": "stage-2-ema-quality-quantity-balance",
        "dyn_oil_th": float(oil_prob_th),
        "dyn_dry_th": float(dry_prob_th),
        "raw_oil": int(raw_oil_count),
        "raw_dry": int(raw_dry_count),
        "topk_oil_current": int(len(next_oil)),
        "topk_dry_current": int(len(next_dry)),
        "pseudo_oil": int(len(next_oil)),
        "pseudo_dry": int(len(next_dry)),
        "eff_pos_neg_ratio": float(eff_pos / max(eff_neg, 1)),
        "max_unlab_prob": max_prob_dbg,
        "p95_unlab_prob": p95_prob_dbg,
        "mean_unlab_std": mean_std_dbg,
        "oil_relaxed": 0,
        "dry_forced": 0,
        "oil_allow": int(raw_oil_count),
        "dry_allow": int(raw_dry_count),
        "oil_allow_cap": int(oil_cap),
        "oil_allow_ratio": int(len(next_oil)),
        "dry_allow_cap": int(dry_cap),
        "dry_allow_ratio": int(len(next_dry)),
        "pruned_oil": int(max(raw_oil_count - len(next_oil), 0)),
        "pruned_dry": int(max(raw_dry_count - len(next_dry), 0)),
        "proj_oil_pass": 0,
        "proj_sim_mean": float("nan"),
        "proj_sim_p95": float("nan"),
        "proj_margin_mean": float("nan"),
        "adaptive_conf_threshold": float(global_thr),
        "softmatch_weight_mean": float(q_mean),
        "csl_score_quantile": float("nan"),
        "csl_oil_score_threshold": float(oil_prob_th),
        "csl_dry_score_threshold": float(dry_prob_th),
        "class_ratio_guard_low": float(cfg.get("ema_min_class_ratio", 0.10)),
        "class_ratio_guard_hi": float(cfg.get("ema_max_class_ratio", 0.90)),
        "pre_guard_oil": int(raw_oil_count),
        "pre_guard_dry": int(raw_dry_count),
        "scheduled_oil_cap": int(oil_cap),
        "scheduled_dry_cap": int(dry_cap),
        "balance_pseudo_pool_capacity": int(True),
        "force_final_pos_neg_1to1": int(True),
        "ema_global_conf": float(ema_global_conf),
        "ema_oil_conf": float(ema_oil_conf),
        "ema_dry_conf": float(ema_dry_conf),
        "ema_pred_oil_ratio": float(ema_pred_oil_ratio),
        "ema_std": float(ema_std),
        "std_gate": float(std_gate),
        "target_oil_ratio": float(target_oil_ratio),
        "pseudo_quality_mean": float(q_mean),
        "total_pseudo_cap": int(total_cap),
    }


def build_train_batch(
    base_trn_indices: np.ndarray,
    base_trn_labels: np.ndarray,
    pseudo_oil_dict: Dict[int, float],
    pseudo_dry_set: set,
    device,
    use_pseudo: bool,
):
    base_oil_idx = base_trn_indices[base_trn_labels == 1]
    base_dry_idx = base_trn_indices[base_trn_labels == 0]

    if use_pseudo:
        pseudo_oil_idx = np.array(sorted(list(pseudo_oil_dict.keys())), dtype=np.int64)
        pseudo_oil_values = np.ones(len(pseudo_oil_idx), dtype=np.float32)
        pseudo_oil_quality = np.array(
            [float(pseudo_oil_dict.get(int(idx), 1.0)) for idx in pseudo_oil_idx],
            dtype=np.float32,
        ) if len(pseudo_oil_idx) > 0 else np.array([], dtype=np.float32)


        if isinstance(pseudo_dry_set, dict):
            pseudo_dry_idx = np.array(sorted(list(pseudo_dry_set.keys())), dtype=np.int64)
            pseudo_dry_quality = np.array(
                [float(pseudo_dry_set.get(int(idx), 1.0)) for idx in pseudo_dry_idx],
                dtype=np.float32,
            ) if len(pseudo_dry_idx) > 0 else np.array([], dtype=np.float32)
        else:
            pseudo_dry_idx = np.array(sorted(list(pseudo_dry_set)), dtype=np.int64)
            pseudo_dry_quality = np.ones(len(pseudo_dry_idx), dtype=np.float32)
    else:
        pseudo_oil_idx = np.array([], dtype=np.int64)
        pseudo_oil_values = np.array([], dtype=np.float32)
        pseudo_oil_quality = np.array([], dtype=np.float32)
        pseudo_dry_idx = np.array([], dtype=np.int64)
        pseudo_dry_quality = np.array([], dtype=np.float32)

    pseudo_oil_quality = np.clip(np.nan_to_num(pseudo_oil_quality, nan=1.0, posinf=1.0, neginf=0.25), 0.05, 1.0)
    pseudo_dry_quality = np.clip(np.nan_to_num(pseudo_dry_quality, nan=1.0, posinf=1.0, neginf=0.25), 0.05, 1.0)

    if len(pseudo_oil_idx) > 0:
        all_oil_idx = np.concatenate([base_oil_idx, pseudo_oil_idx])
    else:
        all_oil_idx = base_oil_idx

    if len(pseudo_dry_idx) > 0:
        all_dry_idx = np.concatenate([base_dry_idx, pseudo_dry_idx])
    else:
        all_dry_idx = base_dry_idx

    oil_labels = torch.cat(
        [
            torch.ones(len(base_oil_idx), 1, device=device),
            torch.FloatTensor(pseudo_oil_values).unsqueeze(1).to(device)
            if len(pseudo_oil_idx) > 0
            else torch.empty(0, 1, device=device),
        ],
        dim=0,
    )

    dry_labels = torch.zeros(len(all_dry_idx), 1, device=device)

    oil_weights = torch.cat(
        [
            torch.full((len(base_oil_idx), 1), float(REAL_OIL_WEIGHT), device=device),
            torch.FloatTensor(float(PSEUDO_CFG["pseudo_oil_weight"]) * pseudo_oil_quality).unsqueeze(1).to(device)
            if len(pseudo_oil_idx) > 0
            else torch.empty(0, 1, device=device),
        ],
        dim=0,
    )

    dry_weights = torch.cat(
        [
            torch.full((len(base_dry_idx), 1), float(REAL_DRY_WEIGHT), device=device),
            torch.FloatTensor(float(PSEUDO_CFG["pseudo_dry_weight"]) * pseudo_dry_quality).unsqueeze(1).to(device)
            if len(pseudo_dry_idx) > 0
            else torch.empty(0, 1, device=device),
        ],
        dim=0,
    )



    if len(all_oil_idx) > 0 and len(all_dry_idx) > 0:
        target_n = int(max(len(all_oil_idx), len(all_dry_idx)))

        if len(all_oil_idx) < target_n:
            extra_pos = np.random.choice(np.arange(len(all_oil_idx)), size=target_n - len(all_oil_idx), replace=True)
            extra_pos_t = torch.LongTensor(extra_pos).to(device)
            all_oil_idx = np.concatenate([all_oil_idx, all_oil_idx[extra_pos]])
            oil_labels = torch.cat([oil_labels, oil_labels[extra_pos_t]], dim=0)
            oil_weights = torch.cat([oil_weights, oil_weights[extra_pos_t]], dim=0)

        if len(all_dry_idx) < target_n:
            extra_neg = np.random.choice(np.arange(len(all_dry_idx)), size=target_n - len(all_dry_idx), replace=True)
            extra_neg_t = torch.LongTensor(extra_neg).to(device)
            all_dry_idx = np.concatenate([all_dry_idx, all_dry_idx[extra_neg]])
            dry_labels = torch.cat([dry_labels, dry_labels[extra_neg_t]], dim=0)
            dry_weights = torch.cat([dry_weights, dry_weights[extra_neg_t]], dim=0)

    batch_indices = np.concatenate([all_oil_idx, all_dry_idx])
    batch_labels = torch.cat([oil_labels, dry_labels], dim=0)
    batch_weights = torch.cat([oil_weights, dry_weights], dim=0)

    if batch_indices.size > 1:
        perm_np = np.random.permutation(batch_indices.size)
        perm_t = torch.LongTensor(perm_np).to(device)
        batch_indices = batch_indices[perm_np]
        batch_labels = batch_labels[perm_t]
        batch_weights = batch_weights[perm_t]

    batch_indices_t = torch.LongTensor(batch_indices).to(device)

    train_pos_count = int(len(all_oil_idx))
    train_neg_count = int(len(all_dry_idx))

    pos_neg_ratio = float(train_pos_count / max(train_neg_count, 1))

    return {
        "batch_indices_t": batch_indices_t,
        "batch_labels": batch_labels,
        "batch_weights": batch_weights,
        "train_pos_count": train_pos_count,
        "train_neg_count": train_neg_count,
        "pos_neg_ratio": pos_neg_ratio,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    x_data: torch.Tensor,
    coords: torch.Tensor,
    val_indices: np.ndarray,
    val_labels_np: np.ndarray,
    device,
):
    model.eval()

    val_indices_t = torch.LongTensor(val_indices).to(device)

    logits = model(
        x_data,
        coords,
        inject_noise=False,
        return_aux=False,
    )[val_indices_t]

    labels_t = torch.FloatTensor(val_labels_np).unsqueeze(1).to(device)

    val_weights = torch.where(
        labels_t > 0.5,
        torch.full_like(labels_t, float(REAL_OIL_WEIGHT)),
        torch.full_like(labels_t, float(REAL_DRY_WEIGHT)),
    )
    val_loss = weighted_focal_loss_with_logits(logits, labels_t, val_weights).item()

    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(val_labels_np, preds)
    per = precision_score(val_labels_np, preds, zero_division=0)
    rec = recall_score(val_labels_np, preds, zero_division=0)
    f1 = f1_score(val_labels_np, preds, zero_division=0)

    auc_v, fpr_v = compute_auc_fpr(
        val_labels_np,
        probs,
        threshold=0.5,
    )

    return {
        "val_loss": float(val_loss),
        "acc": float(acc),
        "per": float(per),
        "rec": float(rec),
        "f1": float(f1),
        "auc": float(auc_v),
        "fpr": float(fpr_v),
    }


def save_final_prediction_map(
    prob_map: np.ndarray,
    data_bundle: Dict,
    out_dir: str,
    filename: str = "最终预测结果图.png",
    title: str = "最终预测概率图（工区内 P>0.5）",
    verbose: bool = True,
):

    x_grid = data_bundle["x_grid"]
    y_grid = data_bundle["y_grid"]
    well_df = data_bundle["well_df"]

    work_area_mask = data_bundle.get("work_area_mask", np.ones_like(prob_map, dtype=bool))
    work_boundary_xy = data_bundle.get("work_boundary_xy", None)
    reserve_boundary_xy = data_bundle.get("reserve_boundary_xy", None)
    threshold = float(data_bundle.get("reserve_prob_threshold", RESERVE_PROB_THRESHOLD))

    prob_map = np.asarray(prob_map, dtype=float)
    work_area_mask = np.asarray(work_area_mask, dtype=bool)

    display_mask = work_area_mask & np.isfinite(prob_map) & (prob_map > threshold)
    prob_display = np.where(display_mask, prob_map, np.nan)
    prob_display = np.ma.masked_invalid(prob_display)

    save_path = os.path.join(out_dir, filename)

    fig, ax = plt.subplots(figsize=(8.8, 9.2), dpi=300)
    ax.set_facecolor("white")

    im = None
    if np.any(display_mask):
        levels = np.linspace(threshold, 1.0, 6)
        im = ax.contourf(
            x_grid,
            y_grid,
            prob_display,
            levels=levels,
            vmin=threshold,
            vmax=1.0,
            cmap="YlOrRd",
            antialiased=False,
            zorder=1,
        )


        ax.contour(
            x_grid,
            y_grid,
            display_mask.astype(float),
            levels=[0.5],
            colors="dimgray",
            linewidths=0.65,
            zorder=7,
        )
    else:
        ax.text(
            0.5,
            0.5,
            f"工区内部无 P>{threshold:.1f} 的预测区域",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="black",
            zorder=2,
        )

    if has_boundary(reserve_boundary_xy, min_points=2):
        plot_boundary_segments(
            ax=ax,
            boundary_obj=reserve_boundary_xy,
            color="blue",
            linewidth=1.6,
            label="储量区边界",
            zorder=11,
        )

    draw_work_area_boundary(
        ax=ax,
        x_grid=x_grid,
        y_grid=y_grid,
        work_area_mask=work_area_mask,
        boundary_xy=work_boundary_xy,
    )

    oil_df = well_df[well_df["label"] == 1].copy()
    dry_df = well_df[well_df["label"] == 0].copy()


    if len(oil_df) > 0:
        inside = mask_points_by_nearest_grid(
            oil_df["X"].values,
            oil_df["Y"].values,
            x_grid,
            y_grid,
            work_area_mask,
        )
        oil_df = oil_df.loc[inside]

    if len(dry_df) > 0:
        inside = mask_points_by_nearest_grid(
            dry_df["X"].values,
            dry_df["Y"].values,
            x_grid,
            y_grid,
            work_area_mask,
        )
        dry_df = dry_df.loc[inside]

    if len(oil_df) > 0:
        ax.scatter(
            oil_df["X"],
            oil_df["Y"],
            s=18,
            c="lime",
            marker="o",
            edgecolors="black",
            linewidths=0.45,
            label="油井",
            zorder=12,
        )

    if len(dry_df) > 0:
        ax.scatter(
            dry_df["X"],
            dry_df["Y"],
            s=18,
            c="black",
            marker="x",
            linewidths=0.80,
            label="干井",
            zorder=13,
        )

    if im is not None:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("预测含油概率")
        cbar.set_ticks(np.linspace(threshold, 1.0, 6))

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")

    handles, labels = ax.get_legend_handles_labels()

    keep_order = ["储量区边界", "工区边界", "油井", "干井"]
    label_to_handle = {}
    for h, lab in zip(handles, labels):
        if lab in keep_order and lab not in label_to_handle:
            label_to_handle[lab] = h
    final_handles = [label_to_handle[k] for k in keep_order if k in label_to_handle]
    final_labels = [k for k in keep_order if k in label_to_handle]
    if final_handles:
        ax.legend(
            final_handles,
            final_labels,
            loc="upper right",
            frameon=True,
            fontsize=8,
            handlelength=2.4,
            borderpad=0.4,
            labelspacing=0.35,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"🖼️ 预测结果图已保存：{save_path}")


def save_epoch_prediction_map(
    model: nn.Module,
    x_data: torch.Tensor,
    coords: torch.Tensor,
    data_bundle: Dict,
    out_dir: str,
    epoch: int,
):

    epoch_dir = os.path.join(out_dir, EPOCH_PREDICTION_DIRNAME)
    os.makedirs(epoch_dir, exist_ok=True)

    if bool(globals().get("INITIAL_EPOCH_PROB_AS_NAN", False)) and epoch < int(globals().get("INITIAL_NAN_EPOCHS", 1)):


        full_probs = np.full((x_data.shape[0],), np.nan, dtype=float)
    else:
        model.eval()
        with torch.no_grad():
            full_logits = model(
                x_data,
                coords,
                inject_noise=False,
                return_aux=False,
            )
            full_probs = torch.sigmoid(full_logits).detach().cpu().numpy().reshape(-1)

    prob_map = full_probs.reshape(data_bundle["grid_res"], data_bundle["grid_res"])
    save_final_prediction_map(
        prob_map=prob_map,
        data_bundle=data_bundle,
        out_dir=epoch_dir,
        filename=f"epoch_{epoch:03d}.png",
        title=f"第 {epoch:03d} 轮预测概率图（工区内 P>0.5）",
        verbose=False,
    )
    print(f"🖼️ 已保存第 {epoch:03d} 轮预测结果图：{os.path.join(epoch_dir, f'epoch_{epoch:03d}.png')}")


def run_training(data_bundle: Dict, device):
    current_dir = data_bundle["current_dir"]

    out_dir = os.path.join(current_dir, OUTPUT_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)
    if SAVE_EPOCH_PREDICTION_MAPS:
        os.makedirs(os.path.join(out_dir, EPOCH_PREDICTION_DIRNAME), exist_ok=True)

    x_data = data_bundle["x_data"]
    coords = data_bundle["coords"]

    grid_res = data_bundle["grid_res"]

    orig_indices = data_bundle["orig_indices"]
    orig_labels = data_bundle["orig_labels"]

    all_sample_idx = np.arange(len(orig_indices))

    trn_idx, val_idx = train_test_split(
        all_sample_idx,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=orig_labels,
        shuffle=True,
    )

    base_trn_indices = orig_indices[trn_idx]
    base_trn_labels = orig_labels[trn_idx]

    curr_val_indices = orig_indices[val_idx]
    curr_val_labels_np = orig_labels[val_idx].astype(np.int64)

    exclude_indices = np.concatenate(
        [
            base_trn_indices,
            curr_val_indices,
        ]
    )

    unlabeled_pool = np.setdiff1d(
        np.arange(grid_res * grid_res),
        exclude_indices,
    )

    pseudo_start_epoch = max(
        int(WARMUP_EPOCHS),
        int(PSEUDO_CFG.get("stage2_start_epoch", WARMUP_EPOCHS)),
    )


    model = DualBranchAttentionModel(
        feat_dim=data_bundle.get("feat_dim", x_data.shape[1]),
        grid_res=grid_res,
        cnn_dim=128,
        vae_hidden=128,
        vae_latent=32,
        dropout_rate=DROPOUT,
        model_variant=MODEL_VARIANT,
    )
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_score = float("inf")  
    best_epoch = -1
    best_metrics = None
    best_model_state = None
    best_history_row = None
    early_stop_counter = 0

    pseudo_oil_dict = {}
    pseudo_dry_set = {}




    cons_ema_probs = None
    pseudo_ema_probs = None

    if CONS_START_EPOCH <= 0:
        init_mean_probs, _ = mc_teacher_predict(
            model,
            x_data,
            coords,
            mc_times=4,
        )
        cons_ema_probs = init_mean_probs
        pseudo_ema_probs = init_mean_probs.copy()

    pseudo_debug = {
        "stage": "warm-up-real-only",
        "dyn_oil_th": PSEUDO_CFG.get("oil_start", PSEUDO_CFG.get("oil_threshold", 0.85)),
        "dyn_dry_th": PSEUDO_CFG["dry_start"],
        "raw_oil": 0,
        "raw_dry": 0,
        "topk_oil_current": 0,
        "topk_dry_current": 0,
        "pseudo_oil": 0,
        "pseudo_dry": 0,
        "eff_pos_neg_ratio": float(
            np.sum(base_trn_labels == 1) / max(np.sum(base_trn_labels == 0), 1)
        ),
    }

    history = []

    print("\n" + "=" * 100)
    print("🚀 开始训练：第1组参数 + 带权Focal主损失 + EMA质量-数量平衡伪标签 + 保留EMA Consistency + 无特征先验")
    print(
        f"seed={RANDOM_SEED} | "
        f"val_size={VAL_SIZE:.2f} | "
        f"epochs={EPOCHS} | "
        f"cons_start={CONS_START_EPOCH} | "
        f"pseudo_start={pseudo_start_epoch} | "
        f"best_start={BEST_SELECTION_START_EPOCH} | "
        f"early_stop_start={EARLY_STOP_START_EPOCH}"
    )
    print(f"Main Loss: {LOSS_NAME}(alpha={FOCAL_ALPHA:.2f}, gamma={FOCAL_GAMMA:.2f}) | RealW(O/D)={REAL_OIL_WEIGHT:.2f}/{REAL_DRY_WEIGHT:.2f} | PseudoW(O/D)={PSEUDO_CFG.get('pseudo_oil_weight')}/{PSEUDO_CFG.get('pseudo_dry_weight')}")
    print(
        f"Cons 半监督损失: 保留EMA teacher | weight={CONS_WEIGHT:.4f} | start_epoch={CONS_START_EPOCH} | "
        f"ramp={CONS_RAMP_EPOCHS} | conf_margin={CONS_CONF_MARGIN:.2f} | max_samples={CONS_MAX_SAMPLES}"
    )
    print("Feature-prior init: 已完全删除；不再计算/叠加 feature-similarity prior logits，也不再设置 prior bias")
    print("Losses: Weighted Focal + EMA Consistency + weak Smoothness + CVAE Recon/KL；伪标签EMA平滑已启用")
    print("Pseudo labels: Stage1前20轮只用真实井；Stage2使用EMA质量-数量平衡：EMA动态阈值 + 动态总容量 + 质量权重；不再固定0.80/0.20，最终训练Pos/Neg=1:1")
    print("Removed: old CNN front end, contrastive, Bal, Rep; CVAE uses u=mu downstream")
    print(f"Model variant: {MODEL_VARIANT}")
    print("Spatial representation: attributes -> Transformer; coordinate conditioning -> Coord-VAE; region propagation -> SSN-GATv2")
    print(f"Architecture road={MODEL_VARIANT} | Transformer(no coords) d_model={TRANSFORMER_D_MODEL} heads={TRANSFORMER_NHEAD} layers={TRANSFORMER_LAYERS} | CVAE latent={PROJ_DIM} | SSN={SSN_SCALES}")
    print(f"Optimizer: AdamW + main cls backward + auxiliary CAGrad(alpha={CAGRAD_ALPHA:.2f})")
    print("选模方式：依旧 Val Loss 优先；F1/AUC/FPR 仅作为 Val Loss 相同情况下的次级参考")
    print(
        f"Pseudo labels: EMA-Q&Q | stage2_start={pseudo_start_epoch} | "
        f"total_cap={PSEUDO_CFG.get('stage2_total_topk_start')}/{PSEUDO_CFG.get('stage2_total_topk_mid')}/{PSEUDO_CFG.get('stage2_total_topk_end')} | "
        "EMA-smoothed dynamic class thresholds; quality-weighted pseudo loss; final training Pos/Neg=1:1"
    )
    print("Metrics: ACC | PER | REC | F1 | AUC | FPR")
    if SAVE_EPOCH_PREDICTION_MAPS:
        print(f"每轮预测图：已开启 | 保存目录={os.path.join(out_dir, EPOCH_PREDICTION_DIRNAME)}")
    print("=" * 100)

    for epoch in range(EPOCHS):
        model.train()

        optimizer.zero_grad(set_to_none=True)

        batch = build_train_batch(
            base_trn_indices=base_trn_indices,
            base_trn_labels=base_trn_labels,
            pseudo_oil_dict=pseudo_oil_dict,
            pseudo_dry_set=pseudo_dry_set,
            device=device,
            use_pseudo=epoch >= pseudo_start_epoch,
        )

        logits, aux_train = model(
            x_data,
            coords,
            inject_noise=False,
            return_aux=True,
        )

        cls_loss = weighted_focal_loss_with_logits(
            logits[batch["batch_indices_t"]],
            batch["batch_labels"],
            batch["batch_weights"],
        )

        cons_raw_loss = consistency_loss_with_ema_teacher(
            logits=logits,
            ema_probs_np=cons_ema_probs,
            candidate_indices_np=unlabeled_pool,
            conf_margin=CONS_CONF_MARGIN,
            max_samples=CONS_MAX_SAMPLES,
        )
        if epoch >= CONS_START_EPOCH:
            cons_ramp = min(1.0, float(epoch - CONS_START_EPOCH + 1) / max(float(CONS_RAMP_EPOCHS), 1.0))
            cons_weight = float(CONS_WEIGHT) * cons_ramp
        else:
            cons_weight = 0.0

        weighted_cons_loss = float(cons_weight) * cons_raw_loss
        smooth_raw_loss = spatial_smoothness_loss(
            logits=logits,
            grid_res=grid_res,
            work_area_mask_np=data_bundle.get("work_area_mask", None),
        )
        weighted_smooth_loss = float(SMOOTH_WEIGHT) * smooth_raw_loss
        recon_raw_loss = reconstruction_loss(
            recon=aux_train.get("recon"),
            target_feat=aux_train.get("transformer_feat"),
        )
        kl_raw_loss = kl_divergence_loss(
            u=aux_train.get("mu"),
            logvar=aux_train.get("logvar"),
        )
        weighted_recon_loss = float(RECON_WEIGHT) * recon_raw_loss
        weighted_kl_loss = float(KL_WEIGHT) * kl_raw_loss

        loss = cls_loss + weighted_cons_loss + weighted_smooth_loss + weighted_recon_loss + weighted_kl_loss
        loss = finite_scalar_loss(loss, device=device)




        cagrad_info = cagrad_backward(
            {
                "cons": weighted_cons_loss,
                "smooth": weighted_smooth_loss,
                "recon": weighted_recon_loss,
                "kl": weighted_kl_loss,
            },
            model=model,
            alpha=CAGRAD_ALPHA,
            rescale=CAGRAD_RESCALE,
            retain_graph_for_later_backward=True,
        )

        cls_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=3.0,
        )

        optimizer.step()

        teacher_start_epoch = min(int(CONS_START_EPOCH), int(pseudo_start_epoch))
        if epoch >= teacher_start_epoch:
            mean_probs, std_probs = mc_teacher_predict(
                model,
                x_data,
                coords,
                mc_times=4,
            )


            if cons_ema_probs is None:
                cons_ema_probs = mean_probs
            else:
                cons_ema_probs = 0.70 * cons_ema_probs + 0.30 * mean_probs



            pseudo_momentum = float(PSEUDO_CFG.get("ema_momentum", 0.90))
            if pseudo_ema_probs is None:
                pseudo_ema_probs = mean_probs.copy()
            else:
                pseudo_ema_probs = pseudo_momentum * pseudo_ema_probs + (1.0 - pseudo_momentum) * mean_probs

            if epoch >= pseudo_start_epoch:


                pseudo_oil_dict, pseudo_dry_set, pseudo_debug = update_pseudo_labels(
                    ema_probs=pseudo_ema_probs,
                    std_probs=std_probs,
                    unlabeled_pool=unlabeled_pool,
                    base_trn_labels=base_trn_labels,
                    pseudo_oil_dict=pseudo_oil_dict,
                    pseudo_dry_set=pseudo_dry_set,
                    epoch=epoch,
                    total_epochs=EPOCHS,
                    warmup_epochs=WARMUP_EPOCHS,
                    cfg=PSEUDO_CFG,
                    proj_emb=None,
                    real_oil_indices=None,
                    real_dry_indices=None,
                )

        metrics = evaluate_model(
            model=model,
            x_data=x_data,
            coords=coords,
            val_indices=curr_val_indices,
            val_labels_np=curr_val_labels_np,
            device=device,
        )

        current_train_pos_count = int(np.sum(base_trn_labels == 1) + len(pseudo_oil_dict))
        current_train_neg_count = int(np.sum(base_trn_labels == 0) + len(pseudo_dry_set))
        current_pos_neg_ratio = float(current_train_pos_count / max(current_train_neg_count, 1))
        phase_name = pseudo_debug.get("stage", "")
        if epoch < pseudo_start_epoch:
            phase_name = "real-strong-before-progressive-pseudo"

        branch_attn_train = aux_train.get("branch_attn", None)
        if torch.is_tensor(branch_attn_train) and branch_attn_train.ndim == 2 and branch_attn_train.shape[1] >= 2:
            fb_front_mean = safe_float(branch_attn_train[:, 0])
            fb_backend_mean = safe_float(branch_attn_train[:, 1])
        else:
            fb_front_mean = float("nan")
            fb_backend_mean = float("nan")
        fb_front_backend_ratio = (
            float(fb_front_mean) / max(float(fb_backend_mean), 1e-8)
            if np.isfinite(fb_front_mean) and np.isfinite(fb_backend_mean)
            else float("nan")
        )

        row = {
            "epoch": int(epoch),
            "loss": float(safe_float(loss)),
            "cls_loss": float(safe_float(cls_loss)),
            "cons_loss": float(safe_float(cons_raw_loss)),
            "cons_weight": float(cons_weight),
            "smooth_loss": float(safe_float(smooth_raw_loss)),
            "smooth_weight": float(SMOOTH_WEIGHT),
            "recon_loss": float(safe_float(recon_raw_loss)),
            "recon_weight": float(RECON_WEIGHT),
            "kl_loss": float(safe_float(kl_raw_loss)),
            "kl_weight": float(KL_WEIGHT),
            "cagrad_active": int(cagrad_info.get("cagrad_active", 0)),
            "cagrad_weights": str(cagrad_info.get("cagrad_weights", "")),
            "fb_front_weight": float(fb_front_mean),
            "fb_backend_weight": float(fb_backend_mean),
            "fb_front_backend_ratio": float(fb_front_backend_ratio),
            "val_loss": float(metrics["val_loss"]),
            "acc": float(metrics["acc"]),
            "per": float(metrics["per"]),
            "rec": float(metrics["rec"]),
            "f1": float(metrics["f1"]),
            "auc": float(metrics["auc"]),
            "fpr": float(metrics["fpr"]),
            "phase": phase_name,
            "raw_oil": int(pseudo_debug.get("raw_oil", 0)),
            "raw_dry": int(pseudo_debug.get("raw_dry", 0)),
            "oil_pool": int(len(pseudo_oil_dict)),
            "dry_pool": int(len(pseudo_dry_set)),
            "topk_oil": int(pseudo_debug.get("topk_oil_current", 0)),
            "topk_dry": int(pseudo_debug.get("topk_dry_current", 0)),
            "oil_threshold": float(
                pseudo_debug.get("dyn_oil_th", PSEUDO_CFG.get("oil_start", PSEUDO_CFG.get("oil_threshold", 0.85)))
            ),
            "dry_threshold": float(
                pseudo_debug.get("dyn_dry_th", PSEUDO_CFG["dry_start"])
            ),
            "train_pos": int(current_train_pos_count),
            "train_neg": int(current_train_neg_count),
            "pos_neg_ratio": float(current_pos_neg_ratio),
            "eff_pos_neg_ratio": float(
                pseudo_debug.get("eff_pos_neg_ratio", current_pos_neg_ratio)
            ),
            "oil_allow": int(pseudo_debug.get("oil_allow", 0)),
            "dry_allow": int(pseudo_debug.get("dry_allow", 0)),
            "oil_allow_cap": int(pseudo_debug.get("oil_allow_cap", 0)),
            "oil_allow_ratio": int(pseudo_debug.get("oil_allow_ratio", 0)),
            "dry_allow_cap": int(pseudo_debug.get("dry_allow_cap", 0)),
            "dry_allow_ratio": int(pseudo_debug.get("dry_allow_ratio", 0)),
            "pruned_oil": int(pseudo_debug.get("pruned_oil", 0)),
            "pruned_dry": int(pseudo_debug.get("pruned_dry", 0)),
            "max_unlab_prob": float(pseudo_debug.get("max_unlab_prob", float("nan"))),
            "p95_unlab_prob": float(pseudo_debug.get("p95_unlab_prob", float("nan"))),
            "mean_unlab_std": float(pseudo_debug.get("mean_unlab_std", float("nan"))),
            "oil_relaxed": int(pseudo_debug.get("oil_relaxed", 0)),
            "dry_forced": int(pseudo_debug.get("dry_forced", 0)),
            "proj_oil_pass": int(pseudo_debug.get("proj_oil_pass", 0)),
            "proj_sim_mean": float(pseudo_debug.get("proj_sim_mean", float("nan"))),
            "proj_sim_p95": float(pseudo_debug.get("proj_sim_p95", float("nan"))),
            "proj_margin_mean": float(pseudo_debug.get("proj_margin_mean", float("nan"))),
            "adaptive_conf_threshold": float(pseudo_debug.get("adaptive_conf_threshold", float("nan"))),
            "softmatch_weight_mean": float(pseudo_debug.get("softmatch_weight_mean", float("nan"))),
            "csl_score_quantile": float(pseudo_debug.get("csl_score_quantile", float("nan"))),
            "csl_oil_score_threshold": float(pseudo_debug.get("csl_oil_score_threshold", float("nan"))),
            "csl_dry_score_threshold": float(pseudo_debug.get("csl_dry_score_threshold", float("nan"))),
            "class_ratio_guard_low": float(pseudo_debug.get("class_ratio_guard_low", float("nan"))),
            "class_ratio_guard_hi": float(pseudo_debug.get("class_ratio_guard_hi", float("nan"))),
            "pre_guard_oil": int(pseudo_debug.get("pre_guard_oil", 0)),
            "pre_guard_dry": int(pseudo_debug.get("pre_guard_dry", 0)),
            "ema_global_conf": float(pseudo_debug.get("ema_global_conf", float("nan"))),
            "ema_oil_conf": float(pseudo_debug.get("ema_oil_conf", float("nan"))),
            "ema_dry_conf": float(pseudo_debug.get("ema_dry_conf", float("nan"))),
            "ema_pred_oil_ratio": float(pseudo_debug.get("ema_pred_oil_ratio", float("nan"))),
            "target_oil_ratio": float(pseudo_debug.get("target_oil_ratio", float("nan"))),
            "std_gate": float(pseudo_debug.get("std_gate", float("nan"))),
            "pseudo_quality_mean": float(pseudo_debug.get("pseudo_quality_mean", float("nan"))),
            "total_pseudo_cap": int(pseudo_debug.get("total_pseudo_cap", 0)),
        }

        history.append(row)



        current_val_loss = float(metrics.get("val_loss", float("inf")))
        current_f1 = float(metrics.get("f1", 0.0))
        current_auc = metrics.get("auc", float("nan"))
        current_auc_safe = 0.0 if not np.isfinite(current_auc) else float(current_auc)
        current_fpr = float(metrics.get("fpr", 1.0))

        val_loss_improved = current_val_loss < best_val_loss - 1e-6
        val_loss_tied = abs(current_val_loss - best_val_loss) <= 1e-6

        if best_metrics is None:
            tie_better = True
        else:
            best_f1_now = float(best_metrics.get("f1", 0.0))
            best_auc_now = best_metrics.get("auc", float("nan"))
            best_auc_now = 0.0 if not np.isfinite(best_auc_now) else float(best_auc_now)
            best_fpr_now = float(best_metrics.get("fpr", 1.0))
            tie_better = (
                (current_f1 > best_f1_now + 1e-10)
                or (
                    abs(current_f1 - best_f1_now) <= 1e-10
                    and current_auc_safe > best_auc_now + 1e-10
                )
                or (
                    abs(current_f1 - best_f1_now) <= 1e-10
                    and abs(current_auc_safe - best_auc_now) <= 1e-10
                    and current_fpr < best_fpr_now - 1e-10
                )
            )

        best_select_allowed = epoch >= int(BEST_SELECTION_START_EPOCH)
        early_stop_allowed = epoch >= int(EARLY_STOP_START_EPOCH)

        if best_select_allowed and (val_loss_improved or (val_loss_tied and tie_better)):
            best_score = float(current_val_loss)
            best_val_loss = float(current_val_loss)
            best_epoch = epoch
            best_metrics = dict(metrics)
            best_model_state = copy.deepcopy(model.state_dict())
            best_history_row = copy.deepcopy(row)
            early_stop_counter = 0
        else:
            if early_stop_allowed and best_model_state is not None:
                early_stop_counter += 1
            else:
                early_stop_counter = 0

        print(
            f"Ep {epoch:03d} | "
            f"Phase={row['phase']} | "
            f"Loss={row['loss']:.4f} | "
            f"Cls={row['cls_loss']:.4f} | "
            f"FB(front/back)={metric_str(row['fb_front_weight'], 3)}/{metric_str(row['fb_backend_weight'], 3)} "
            f"R={metric_str(row['fb_front_backend_ratio'], 3)} | "
            f"Cons={row['cons_loss']:.4f}*{row['cons_weight']:.4f} | "
            f"Smooth={row['smooth_loss']:.4f}*{row['smooth_weight']:.4f} | "
            f"Recon={row['recon_loss']:.4f}*{row['recon_weight']:.4f} | "
            f"KL={row['kl_loss']:.4f}*{row['kl_weight']:.4f} | "
            f"CAGrad={row['cagrad_active']}[{row['cagrad_weights']}] | "
            f"Val={metrics['val_loss']:.4f} | "
            f"ACC={metrics['acc']:.4f} "
            f"PER={metrics['per']:.4f} "
            f"REC={metrics['rec']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"AUC={metric_str(metrics['auc'])} "
            f"FPR={metric_str(metrics['fpr'])} | "
            f"Raw(O/D)={row['raw_oil']}/{row['raw_dry']} | "
            f"OilPool={row['oil_pool']} DryPool={row['dry_pool']} | "
            f"TopK(O/D)={row['topk_oil']}/{row['topk_dry']} | "
            f"Allow(O/D)={row['oil_allow']}/{row['dry_allow']} | "
            f"AllowCap(O/D)={row['oil_allow_cap']}/{row['dry_allow_cap']} | "
            f"AllowRatio(O/D)={row['oil_allow_ratio']}/{row['dry_allow_ratio']} | "
            f"Prune(O/D)={row['pruned_oil']}/{row['pruned_dry']} | "
            f"Pos/Neg={row['train_pos']}/{row['train_neg']}({row['pos_neg_ratio']:.3f}) | "
            f"EffRatio={row['eff_pos_neg_ratio']:.3f} | "
            f"Thr(O/D)={row['oil_threshold']:.3f}/{row['dry_threshold']:.3f} | "
            f"CurConf={metric_str(row['ema_global_conf'], 3)} "
            f"CurRatioO={metric_str(row['ema_pred_oil_ratio'], 3)} "
            f"TargetO={metric_str(row['target_oil_ratio'], 3)} "
            f"Qmean={metric_str(row['pseudo_quality_mean'], 3)} "
            f"StdGate={metric_str(row['std_gate'], 3)} | "
            f"MaxP={metric_str(row['max_unlab_prob'], 3)} "
            f"P95={metric_str(row['p95_unlab_prob'], 3)}"
        )

        if SAVE_EPOCH_PREDICTION_MAPS:
            save_epoch_prediction_map(
                model=model,
                x_data=x_data,
                coords=coords,
                data_bundle=data_bundle,
                out_dir=out_dir,
                epoch=epoch,
            )
            release_cuda_memory()


        if early_stop_counter >= PATIENCE:
            print(f"⏹️ Early Stop at epoch {epoch}")
            break

    if best_model_state is None:
        best_model_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch

        best_metrics = evaluate_model(
            model=model,
            x_data=x_data,
            coords=coords,
            val_indices=curr_val_indices,
            val_labels_np=curr_val_labels_np,
            device=device,
        )
        best_history_row = history[-1] if len(history) > 0 else {"epoch": int(best_epoch)}

    model.load_state_dict(best_model_state)
    model.eval()

    with torch.no_grad():
        full_logits = model(
            x_data,
            coords,
            inject_noise=False,
            return_aux=False,
        )

        full_probs = torch.sigmoid(
            full_logits
        ).detach().cpu().numpy().reshape(-1)

    prob_map = full_probs.reshape(grid_res, grid_res)

    reserve_boundary_xy = data_bundle.get("reserve_boundary_xy", None)
    work_area_mask_for_cover = data_bundle.get("work_area_mask", np.ones_like(prob_map, dtype=bool))
    reserve_coverge = float("nan")
    if has_boundary(reserve_boundary_xy, min_points=3):
        try:
            reserve_area_mask = boundary_union_mask(
                data_bundle["x_grid"],
                data_bundle["y_grid"],
                reserve_boundary_xy,
            ) & work_area_mask_for_cover.astype(bool)
            pred_area_mask = (prob_map > RESERVE_PROB_THRESHOLD) & work_area_mask_for_cover.astype(bool)
            denom = int(np.sum(reserve_area_mask))
            reserve_coverge = float(np.sum(pred_area_mask & reserve_area_mask) / max(denom, 1))
        except Exception:
            reserve_coverge = float("nan")

    history_df = pd.DataFrame(history)

    history_path = os.path.join(out_dir, "训练日志.csv")
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

    work_area_mask_flat = data_bundle.get(
        "work_area_mask",
        np.ones_like(prob_map, dtype=bool),
    ).reshape(-1).astype(bool)
    reserve_threshold = float(data_bundle.get("reserve_prob_threshold", RESERVE_PROB_THRESHOLD))
    display_mask_flat = work_area_mask_flat & np.isfinite(full_probs) & (full_probs > reserve_threshold)

    pred_df = pd.DataFrame(
        {
            "x": data_bundle["x_grid"].reshape(-1),
            "y": data_bundle["y_grid"].reshape(-1),
            "prob": full_probs,
            "inside_work_area": work_area_mask_flat.astype(int),
            "display_gt05_inside_work": display_mask_flat.astype(int),
            "prob_display_gt05_inside_work": np.where(display_mask_flat, full_probs, np.nan),
            "pred_label_05_inside_work": display_mask_flat.astype(int),
        }
    )

    pred_path = os.path.join(out_dir, "最终全区预测概率.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    save_final_prediction_map(
        prob_map=prob_map,
        data_bundle=data_bundle,
        out_dir=out_dir,
        filename="最终预测结果图.png",
    )

    model_path = os.path.join(out_dir, "best_model.pth")
    torch.save(best_model_state, model_path)

    final_history_row = history[-1] if len(history) > 0 else {"epoch": int(best_epoch)}
    pseudo_explanation = build_pseudo_explanation(
        best_row=best_history_row,
        final_row=final_history_row,
        cfg=PSEUDO_CFG,
    )

    pseudo_explain_path = os.path.join(out_dir, "伪标签未加入原因说明.txt")
    with open(pseudo_explain_path, "w", encoding="utf-8") as f:
        f.write(pseudo_explanation + "\n")

    summary = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_select_score": float(best_score),  
        "best_acc": float(best_metrics["acc"]),
        "best_per": float(best_metrics["per"]),
        "best_rec": float(best_metrics["rec"]),
        "best_f1": float(best_metrics["f1"]),
        "best_auc": float(best_metrics["auc"]),
        "best_fpr": float(best_metrics["fpr"]),
        "best_oil_pool": int((best_history_row or {}).get("oil_pool", 0)),
        "best_dry_pool": int((best_history_row or {}).get("dry_pool", 0)),
        "best_raw_oil": int((best_history_row or {}).get("raw_oil", 0)),
        "best_raw_dry": int((best_history_row or {}).get("raw_dry", 0)),
        "best_topk_oil": int((best_history_row or {}).get("topk_oil", 0)),
        "best_topk_dry": int((best_history_row or {}).get("topk_dry", 0)),
        "final_training_oil_pool": int(len(pseudo_oil_dict)),
        "final_training_dry_pool": int(len(pseudo_dry_set)),
        "best_fb_front_weight": float((best_history_row or {}).get("fb_front_weight", float("nan"))),
        "best_fb_backend_weight": float((best_history_row or {}).get("fb_backend_weight", float("nan"))),
        "best_fb_front_backend_ratio": float((best_history_row or {}).get("fb_front_backend_ratio", float("nan"))),
        "feature_prior_removed": True,
        "pseudo_start_epoch": int(pseudo_start_epoch),
        "best_selection_start_epoch": int(BEST_SELECTION_START_EPOCH),
        "early_stop_start_epoch": int(EARLY_STOP_START_EPOCH),
        "reserve_probability_threshold": float(reserve_threshold),
        "COVERGE": float(reserve_coverge),
        "display_rule": "仅显示工区内部 prob > 0.5 的概率值；标准储量区文件蓝线；工区边界黑线",
        "pseudo_explain_txt": pseudo_explain_path,
        "history_csv": history_path,
        "prediction_csv": pred_path,
        "prediction_png": os.path.join(out_dir, "最终预测结果图.png"),
        "model_path": model_path,
    }

    summary_path = os.path.join(out_dir, "最优验证结果.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("✅ 训练完成")
    print(f"Best Epoch={best_epoch}")
    print(
        f"ACC={summary['best_acc']:.4f} | "
        f"PER={summary['best_per']:.4f} | "
        f"REC={summary['best_rec']:.4f} | "
        f"F1={summary['best_f1']:.4f} | "
        f"AUC={metric_str(summary['best_auc'])} | "
        f"FPR={metric_str(summary['best_fpr'])} | "
        f"COVERGE={metric_str(summary.get('COVERGE', float('nan')))}"
    )
    print(
        f"最优轮次伪油井：{summary['best_oil_pool']} | "
        f"最优轮次伪干井：{summary['best_dry_pool']}"
    )
    print(
        f"训练停止时伪油井：{summary['final_training_oil_pool']} | "
        f"训练停止时伪干井：{summary['final_training_dry_pool']}"
    )
    print("\n" + pseudo_explanation)
    print(f"伪标签说明TXT：{pseudo_explain_path}")
    print(f"训练日志CSV：{history_path}")
    print(f"最终预测CSV：{pred_path}")
    print(f"最终预测图：{os.path.join(out_dir, '最终预测结果图.png')}")
    if SAVE_EPOCH_PREDICTION_MAPS:
        print(f"每轮预测图目录：{os.path.join(out_dir, EPOCH_PREDICTION_DIRNAME)}")
    print(f"最优模型：{model_path}")
    print(f"结果目录：{out_dir}")
    print("=" * 100)

    return model, prob_map, summary





def _current_config_row() -> Dict:

    row = {
        "config_name": "group001_weighted_focal_a060_g175_QQ_EMA",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "cons_weight": CONS_WEIGHT,
        "cons_start_epoch": CONS_START_EPOCH,
        "cons_ramp_epochs": CONS_RAMP_EPOCHS,
        "cons_conf_margin": CONS_CONF_MARGIN,
        "smooth_weight": SMOOTH_WEIGHT,
        "recon_weight": RECON_WEIGHT,
        "kl_weight": KL_WEIGHT,
        "loss_name": LOSS_NAME,
        "focal_alpha": FOCAL_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
        "real_oil_weight": REAL_OIL_WEIGHT,
        "real_dry_weight": REAL_DRY_WEIGHT,
        "ssn_scales": "-".join(map(str, SSN_SCALES)),
    }
    for key in [
        "stage2_start_epoch",
        "oil_start",
        "oil_mid",
        "oil_end",
        "dry_start",
        "dry_mid",
        "dry_end",
        "stage2_oil_min_prob",
        "stage2_dry_max_prob",
        "adaptive_threshold",
        "adaptive_threshold_stat",
        "softmatch_weighting",
        "softmatch_sigma_min",
        "csl_score_quantile",
        "csl_min_conf_margin",
        "csl_prob_weight",
        "csl_conf_weight",
        "direct_topk_no_threshold",
        "target_pos_neg_low",
        "target_pos_neg_hi",
        "stage2_topk_start",
        "stage2_topk_mid",
        "stage2_topk_end",
        "stage2_total_topk_start",
        "stage2_total_topk_mid",
        "stage2_total_topk_end",
        "ema_balance_enabled",
        "ema_momentum",
        "ema_min_global_conf",
        "ema_max_global_conf",
        "ema_min_class_conf",
        "ema_class_prior_blend",
        "ema_min_class_ratio",
        "ema_max_class_ratio",
        "pseudo_quality_weight_min",
        "pseudo_quality_power",
        "use_uncertainty_gate",
        "ema_std_multiplier",
        "ema_std_min",
        "ema_std_max",
        "allow_reallocate_unused_capacity",
        "force_final_pos_neg_1to1",
        "stage2_oil_topk_start",
        "stage2_oil_topk_mid",
        "stage2_oil_topk_end",
        "stage2_dry_topk_start",
        "stage2_dry_topk_mid",
        "stage2_dry_topk_end",
        "proj_score_weight",
        "proj_margin_score_weight",
        "use_projection_as_soft_score",
        "uncertainty_penalty",
        "pseudo_oil_weight",
        "pseudo_dry_weight",
        "pseudo_oil_label_min",
        "pseudo_oil_label_max",
    ]:
        if key in PSEUDO_CFG:
            row[f"pseudo_{key}"] = PSEUDO_CFG[key]
    return row


def main():
    global OUTPUT_DIRNAME, PSEUDO_CFG, MODEL_VARIANT

    seed_everything(2026)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"TS-SGNet | ACDR + MSGP + EMA-QB | device={device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    example_dir = os.path.join(script_dir, "example_data")
    current_dir = os.environ.get(
        "TSSGNET_DATA_DIR",
        example_dir if os.path.isdir(example_dir) else script_dir,
    )
    current_dir = os.path.abspath(current_dir)

    config_name = "group001_weighted_focal_a060_g175_QQ_EMA"

    base_output_dirname = OUTPUT_DIRNAME
    out_root = os.path.join(current_dir, base_output_dirname)
    os.makedirs(out_root, exist_ok=True)

    print("\n" + "=" * 100)
    print("📌 当前为 LX 数据集固定 train/val 划分：仅保留第1组参数单版本训练")
    print("版本：第一个文件前端 + 第二个文件SSN-GAT后端 | SSN三尺度=100/400/800")
    print("Model setting: Transformer不输入坐标，仅编码7个地质属性；coords作为CoordCVAE条件与SSN-GAT空间信息")
    print("Training: 带权Focal主损失普通反传；CAGrad 只协调 Consistency / Smoothness / CVAE Recon-KL 辅助损失；已完全删除特征先验")
    print("Pseudo label strategy: Stage2从20轮启动；EMA质量-数量平衡：当前轮动态阈值、动态总容量、伪标签质量权重；不固定0.80/0.20，最终训练Pos/Neg=1:1")
    print("📌 储量区边界只用于蓝线标注和 COVERGE 统计，不参与训练、伪标签筛选或损失约束")
    print(f"📁 第1组+带权Focal结果根目录：{out_root}")
    print(f"📌 数据目录：{current_dir}")
    print(f"📌 固定参数：{config_name} | 仅保留原20组搜索中的第1组，已删除参数搜索和组覆盖逻辑")
    print(f"EPOCHS={EPOCHS} | PATIENCE={PATIENCE}")
    print("=" * 100)

    with open(os.path.join(out_root, "第1组_带权Focal参数配置.json"), "w", encoding="utf-8") as f:
        json.dump(_current_config_row(), f, ensure_ascii=False, indent=2)

    data_bundle = prepare_data(current_dir, device)
    base_pseudo_cfg = copy.deepcopy(PSEUDO_CFG)
    all_rows = []

    for road_i, road in enumerate(EXPERIMENT_ROADS, start=1):
        MODEL_VARIANT = str(road["variant"])
        road_label = str(road["label"])
        road_dirname = str(road["dirname"])


        OUTPUT_DIRNAME = os.path.join(base_output_dirname, road_dirname)
        road_out_dir = os.path.join(current_dir, OUTPUT_DIRNAME)
        os.makedirs(road_out_dir, exist_ok=True)

        PSEUDO_CFG = copy.deepcopy(base_pseudo_cfg)

        seed_everything(2026)

        with open(os.path.join(road_out_dir, "当前精简版配置.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "road_index": road_i,
                    "model_variant": MODEL_VARIANT,
                    "road_label": road_label,
                    "param_source": "group001_weighted_focal_a060_g175_QQ_EMA",
                    "config_name": config_name,
                    "output_dir": road_out_dir,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print("\n" + "=" * 100)
        print(f"▶️ 开始训练精简版：{road_label}")
        print(f"结果目录：{road_out_dir}")
        print(
            f"FeaturePrior=Removed | LR={LR} | WD={WEIGHT_DECAY} | Dropout={DROPOUT} | "
            f"Loss={LOSS_NAME}(alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA}) | RealW(O/D)={REAL_OIL_WEIGHT}/{REAL_DRY_WEIGHT} | SSN={SSN_SCALES} | "
            f"Cons={CONS_WEIGHT}@{CONS_START_EPOCH} ramp={CONS_RAMP_EPOCHS} margin={CONS_CONF_MARGIN}"
        )
        print(
            f"PseudoStart={PSEUDO_CFG.get('stage2_start_epoch')} | "
            f"EMA-Q&Q total_cap={PSEUDO_CFG.get('stage2_total_topk_start')}/{PSEUDO_CFG.get('stage2_total_topk_mid')}/{PSEUDO_CFG.get('stage2_total_topk_end')} | "
            f"QualityMin={PSEUDO_CFG.get('pseudo_quality_weight_min')} | "
            f"OilW={PSEUDO_CFG.get('pseudo_oil_weight')} DryW={PSEUDO_CFG.get('pseudo_dry_weight')}"
        )
        print("=" * 100)

        model, prob_map, summary = run_training(data_bundle, device)

        row = _current_config_row()
        row.update(summary)
        row["road_index"] = road_i
        row["model_variant"] = MODEL_VARIANT
        row["road_label"] = road_label
        row["group_output_dir"] = road_out_dir
        all_rows.append(row)

        road_summary_path = os.path.join(road_out_dir, "单版本训练汇总.csv")
        pd.DataFrame([row]).to_csv(road_summary_path, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 100)
        print(f"✅ 精简版训练完成：{road_label}")
        print(
            f"Best Epoch={row.get('best_epoch')} | "
            f"ACC={float(row.get('best_acc')):.4f} | "
            f"PER={float(row.get('best_per')):.4f} | "
            f"REC={float(row.get('best_rec')):.4f} | "
            f"F1={float(row.get('best_f1')):.4f} | "
            f"AUC={metric_str(row.get('best_auc'))} | "
            f"FPR={metric_str(row.get('best_fpr'))} | "
            f"COVERGE={metric_str(row.get('COVERGE'))}"
        )
        print(f"精简版训练汇总：{road_summary_path}")
        print(f"精简版结果目录：{road_out_dir}")
        print("=" * 100)

        del model, prob_map
        release_cuda_memory()

    all_summary_path = os.path.join(out_root, "精简版训练汇总.csv")
    pd.DataFrame(all_rows).to_csv(all_summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("✅ 精简版 front_backend 单版本训练完成")
    print(f"总汇总CSV：{all_summary_path}")
    print(f"精简版结果根目录：{out_root}")
    for row in all_rows:
        print(
            f"{row.get('road_index')}. {row.get('model_variant')} | "
            f"F1={float(row.get('best_f1')):.4f} | "
            f"AUC={metric_str(row.get('best_auc'))} | "
            f"COVERGE={metric_str(row.get('COVERGE'))} | "
            f"dir={row.get('group_output_dir')}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()