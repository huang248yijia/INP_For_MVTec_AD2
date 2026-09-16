import argparse
import csv
import os
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
# ============================================================
# [2026-09-07 修改開始] MVTec AD 2 private submission imports
# private test 沒有 GT/label，因此新增 PIL 與 Dataset 供 private-only dataset 使用。
# ============================================================
from PIL import Image
from torch.utils.data import DataLoader, Dataset
# ============================================================
# [2026-09-07 修改結束] MVTec AD 2 private submission imports
# ============================================================
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from optimizers import StableAdamW
from utils import (
    WarmCosineScheduler,
    cal_anomaly_maps,
    get_gaussian_kernel,
    global_cosine_hm_adaptive,
    get_logger,
    setup_seed,
)

# Dataset
from dataset import MVTecDataset, get_data_transforms

# Model
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import (
    Mlp,
    Aggregation_Block,
    Prototype_Block,
)


warnings.filterwarnings("ignore")


MVTEC_AD2_CATEGORIES = [
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
]


# ============================================================
# Data
# ============================================================

def build_transforms(args):
    """
    這裡只負責呼叫 dataset.py 的 transform。

    你前面已修改過的 dataset.py 應符合：
      1. MVTec AD 2 使用接近 256 且為 14 倍數的輸入尺寸
      2. 不做 CenterCrop
      3. GT mask 使用 nearest-neighbor resize
    """
    if args.input_size % 14 != 0:
        raise ValueError(
            f"input_size={args.input_size} 不是 ViT/14 patch size 的倍數。"
        )

    if args.crop_size != args.input_size:
        raise ValueError(
            "MVTec AD 2 版本請讓 crop_size == input_size，且 dataset.py 不應再做 CenterCrop。"
        )

    return get_data_transforms(
        args.input_size,
        args.crop_size,
    )


def build_train_loader(args, data_transform):
    category_root = os.path.join(
        args.data_path,
        args.item,
    )

    train_path = os.path.join(
        category_root,
        "train",
    )

    if not os.path.isdir(train_path):
        raise FileNotFoundError(
            f"找不到 training folder: {train_path}"
        )

    train_data = ImageFolder(
        root=train_path,
        transform=data_transform,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    return train_data, train_loader


def build_test_public_loader(
    args,
    data_transform,
    gt_transform,
):
    category_root = os.path.join(
        args.data_path,
        args.item,
    )

    test_public_path = os.path.join(
        category_root,
        "test_public",
    )

    if not os.path.isdir(test_public_path):
        raise FileNotFoundError(
            f"找不到 TESTpub folder: {test_public_path}"
        )

    test_data = MVTecDataset(
        root=category_root,
        transform=data_transform,
        gt_transform=gt_transform,
        phase="test_public",
    )

    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return test_data, test_loader


# ============================================================
# [2026-09-07 修改開始] MVTec AD 2 private dataset / loader
#
# 說明：
#   - test_private / test_private_mixed 沒有公開 GT 與 label。
#   - 不修改既有 dataset.py，避免影響 train / validation / test_public。
#   - private inference 僅回傳 (image, image_path)。
# ============================================================

class MVTecAD2PrivateDataset(Dataset):
    def __init__(
        self,
        category_root,
        split_name,
        transform,
    ):
        if split_name not in {
            "test_private",
            "test_private_mixed",
        }:
            raise ValueError(
                f"Unsupported private split: {split_name}"
            )

        self.split_root = os.path.join(
            category_root,
            split_name,
        )

        if not os.path.isdir(self.split_root):
            raise FileNotFoundError(
                f"Private directory does not exist: "
                f"{self.split_root}"
            )

        self.transform = transform
        self.image_paths = []

        valid_exts = {
            ".png", ".PNG",
            ".jpg", ".JPG",
            ".jpeg", ".JPEG",
            ".bmp", ".BMP",
            ".tif", ".TIF",
            ".tiff", ".TIFF",
        }

        # 遞迴搜尋是為了容忍本地資料夾多一層整理結構；
        # 輸出時仍依官方 submission 結構放進 split folder。
        for root, _, files in os.walk(self.split_root):
            for file_name in files:
                ext = os.path.splitext(file_name)[1]
                if ext in valid_exts:
                    self.image_paths.append(
                        os.path.join(root, file_name)
                    )

        self.image_paths = sorted(self.image_paths)

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No private images found in {self.split_root}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, img_path


def build_private_loader(
    args,
    data_transform,
    split_name,
):
    category_root = os.path.join(
        args.data_path,
        args.item,
    )

    private_data = MVTecAD2PrivateDataset(
        category_root=category_root,
        split_name=split_name,
        transform=data_transform,
    )

    private_loader = DataLoader(
        private_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return private_data, private_loader


# ============================================================
# [2026-09-07 修改結束] MVTec AD 2 private dataset / loader
# ============================================================


def build_validation_loader(
    args,
    data_transform,
    gt_transform,
):
    category_root = os.path.join(
        args.data_path,
        args.item,
    )

    validation_path = os.path.join(
        category_root,
        "validation",
    )

    if not os.path.isdir(validation_path):
        raise FileNotFoundError(
            f"找不到 validation folder: {validation_path}"
        )

    val_data = MVTecDataset(
        root=category_root,
        transform=data_transform,
        gt_transform=gt_transform,
        phase="validation",
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return val_data, val_loader


# ============================================================
# Model
# ============================================================

def build_model(args):
    """
    保留原 INP-Former single-class 架構。
    """
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]
    fuse_layer_decoder = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]

    encoder = vit_encoder.load(args.encoder)

    if "small" in args.encoder:
        embed_dim, num_heads = 384, 6

    elif "base" in args.encoder:
        embed_dim, num_heads = 768, 12

    elif "large" in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [
            4, 6, 8, 10, 12, 14, 16, 18
        ]

    else:
        raise ValueError(
            f"不支援的 encoder: {args.encoder}"
        )

    bottleneck = nn.ModuleList([
        Mlp(
            embed_dim,
            embed_dim * 4,
            embed_dim,
            drop=0.0,
        )
    ])

    inp = nn.ParameterList([
        nn.Parameter(
            torch.randn(
                args.INP_num,
                embed_dim,
            )
        )
    ])

    inp_extractor = nn.ModuleList([
        Aggregation_Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(
                nn.LayerNorm,
                eps=1e-8,
            ),
        )
    ])

    inp_guided_decoder = nn.ModuleList([
        Prototype_Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(
                nn.LayerNorm,
                eps=1e-8,
            ),
        )
        for _ in range(8)
    ])

    model = INP_Former(
        encoder=encoder,
        bottleneck=bottleneck,
        aggregation=inp_extractor,
        decoder=inp_guided_decoder,
        target_layers=target_layers,
        remove_class_token=True,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        prototype_token=inp,
    )

    trainable = nn.ModuleList([
        bottleneck,
        inp_guided_decoder,
        inp_extractor,
        inp,
    ])

    return model, trainable


def initialize_trainable(trainable):
    """
    保留原本 initialization。
    """
    for module in trainable.modules():

        if isinstance(module, nn.Linear):
            trunc_normal_(
                module.weight,
                std=0.01,
                a=-0.03,
                b=0.03,
            )

            if module.bias is not None:
                nn.init.constant_(
                    module.bias,
                    0,
                )

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(
                module.bias,
                0,
            )

            nn.init.constant_(
                module.weight,
                1.0,
            )


def checkpoint_path(args):
    return os.path.join(
        args.save_dir,
        args.save_name,
        args.item,
        "model.pth",
    )


# ============================================================
# Train
# ============================================================

def train_one_category(
    args,
    model,
    trainable,
    train_data,
    train_loader,
    device,
):
    """
    保留原始 INP-Former training logic：
      StableAdamW
      lr=1e-3
      weight_decay=1e-4
      WarmCosineScheduler
      global_cosine_hm_adaptive(..., y=3)
      + 0.2 * g_loss
      grad clip = 0.1
    """

    initialize_trainable(
        trainable
    )

    optimizer = StableAdamW(
        [
            {
                "params": (
                    trainable.parameters()
                )
            }
        ],
        lr=1e-3,
        betas=(
            0.9,
            0.999,
        ),
        weight_decay=1e-4,
        amsgrad=True,
        eps=1e-10,
    )

    lr_scheduler = WarmCosineScheduler(
        optimizer,
        base_value=1e-3,
        final_value=1e-4,
        total_iters=(
            args.total_epochs
            * len(train_loader)
        ),
        warmup_iters=100,
    )

    print_fn(
        f"[{args.item}] train images: "
        f"{len(train_data)}"
    )

    for epoch in range(
        args.total_epochs
    ):
        model.train()
        loss_list = []

        for img, _ in tqdm(
            train_loader,
            ncols=90,
            desc=(
                f"{args.item} "
                f"{epoch + 1}/{args.total_epochs}"
            ),
        ):
            img = img.to(
                device,
                non_blocking=True,
            )

            en, de, g_loss = model(img)

            loss = global_cosine_hm_adaptive(
                en,
                de,
                y=3,
            )

            loss = (
                loss
                + 0.2 * g_loss
            )

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(
                trainable.parameters(),
                max_norm=0.1,
            )

            optimizer.step()
            lr_scheduler.step()

            loss_list.append(
                float(
                    loss.item()
                )
            )

        print_fn(
            "epoch [{}/{}], loss:{:.6f}".format(
                epoch + 1,
                args.total_epochs,
                np.mean(
                    loss_list
                ),
            )
        )

    ckpt = checkpoint_path(
        args
    )

    os.makedirs(
        os.path.dirname(
            ckpt
        ),
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        ckpt,
    )

    print_fn(
        f"[{args.item}] checkpoint saved: "
        f"{ckpt}"
    )


def load_checkpoint(
    args,
    model,
    device,
):
    ckpt = checkpoint_path(
        args
    )

    if not os.path.isfile(
        ckpt
    ):
        raise FileNotFoundError(
            f"找不到 checkpoint: {ckpt}"
        )

    state_dict = torch.load(
        ckpt,
        map_location=device,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    print_fn(
        f"[{args.item}] checkpoint loaded: "
        f"{ckpt}"
    )


# ============================================================
# TIFF export
# ============================================================

def get_relative_test_path(
    img_path,
    category_root,
    split_name,
):
    """
    保留原始 TESTpub 相對結構。

    例如：
        .../fabric/test_public/bad/000.png
    會輸出到：
        output/fabric/test_public/bad/000.tiff
    """

    split_root = os.path.join(
        category_root,
        split_name,
    )

    rel = os.path.relpath(
        img_path,
        split_root,
    )

    rel_path = Path(
        rel
    )

    return (
        rel_path.parent,
        rel_path.stem,
    )


def image_score_from_map(
    anomaly_map_2d,
    max_ratio,
):
    """
    完全沿用原 evaluation_batch 的 image score 定義：

    max_ratio == 0:
        max pixel

    max_ratio > 0:
        top max_ratio pixels 的平均
    """

    flat = anomaly_map_2d.reshape(
        -1
    )

    if max_ratio <= 0:
        return float(
            flat.max()
        )

    k = int(
        flat.size
        * max_ratio
    )

    k = max(
        1,
        k,
    )

    if k >= flat.size:
        return float(
            flat.mean()
        )

    topk = np.partition(
        flat,
        flat.size - k,
    )[-k:]

    return float(
        topk.mean()
    )


@torch.no_grad()
def export_original_inp_anomaly_maps(
    args,
    model,
    dataloader,
    device,
    split_name,
):
    
    model.eval()

    category_root = os.path.join(
        args.data_path,
        args.item,
    )

    output_root = os.path.join(
        args.output_dir,
        "anomaly_images_public"
        if split_name == "test_public"
        else "anomaly_images_validation",
        args.item,
        split_name,
    )

    os.makedirs(
        output_root,
        exist_ok=True,
    )

    gaussian_kernel = get_gaussian_kernel(
        kernel_size=5,
        sigma=4,
    ).to(device)

    csv_rows = []
    saved_count = 0

    for (
        img,
        _gt,
        label,
        img_paths,
    ) in tqdm(
        dataloader,
        ncols=90,
        desc=(
            f"Export {args.item}/"
            f"{split_name}"
        ),
    ):
        img = img.to(
            device,
            non_blocking=True,
        )

        output = model(
            img
        )

        en, de = (
            output[0],
            output[1],
        )

        # ----------------------------------------------------
        # 1. 完全沿用原版：
        #    anomaly map 先生成在 img.shape[-1]
        # ----------------------------------------------------
        anomaly_map, _ = cal_anomaly_maps(
            en,
            de,
            img.shape[-1],
        )

        # ----------------------------------------------------
        # 2. 完全沿用原版：
        #    若 resize_mask 有設定，先 resize
        # ----------------------------------------------------
        if args.resize_mask is not None:
            anomaly_map = F.interpolate(
                anomaly_map,
                size=args.resize_mask,
                mode="bilinear",
                align_corners=False,
            )

        # ----------------------------------------------------
        # 3. 完全沿用原版：
        #    resize 完才 Gaussian smoothing
        # ----------------------------------------------------
        anomaly_map = gaussian_kernel(
            anomaly_map
        )

        # ----------------------------------------------------
        # 4. 不再做任何處理，直接存 TIFF
        # ----------------------------------------------------
        batch_size = img.shape[0]

        for b_idx in range(
            batch_size
        ):
            img_path = (
                img_paths[b_idx]
            )

            pred_map = (
                anomaly_map[
                    b_idx
                ]
                .squeeze()
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            rel_parent, stem = (
                get_relative_test_path(
                    img_path,
                    category_root,
                    split_name,
                )
            )

            save_dir = os.path.join(
                output_root,
                str(
                    rel_parent
                ),
            )

            os.makedirs(
                save_dir,
                exist_ok=True,
            )

            save_path = os.path.join(
                save_dir,
                f"{stem}.tiff",
            )

            # raw float32 score
            # 不做 min-max normalization
            tiff.imwrite(
                save_path,
                pred_map,
                dtype=np.float32,
            )

            label_value = int(
                label[b_idx].item()
            )

            csv_rows.append({
                "category": args.item,
                "split": split_name,
                "label": label_value,
                "image_path": os.path.abspath(
                    img_path
                ),
                "anomaly_map_path": os.path.abspath(
                    save_path
                ),
                "map_height": int(
                    pred_map.shape[0]
                ),
                "map_width": int(
                    pred_map.shape[1]
                ),
                "image_score_top_ratio": (
                    image_score_from_map(
                        pred_map,
                        args.max_ratio,
                    )
                ),
                "image_score_max": float(
                    pred_map.max()
                ),
            })

            saved_count += 1

    csv_path = os.path.join(
        output_root,
        "image_scores.csv",
    )

    if len(
        csv_rows
    ) > 0:
        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    csv_rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                csv_rows
            )

    print_fn(
        f"[{args.item}] {split_name}: "
        f"saved {saved_count} TIFF files to "
        f"{output_root}"
    )

    return saved_count


# ============================================================
# [2026-09-07 修改開始] MVTec AD 2 private TIFF export
#
# 官方 submission 要求：
#   1. anomaly_images/<category>/<split>/<filename>.tiff
#   2. non-thresholded continuous anomaly maps
#   3. TIFF dtype = float16
#
# 此函式保留原 INP-Former anomaly-map pipeline：
#   cal_anomaly_maps -> optional bilinear resize -> Gaussian smoothing
# 不做 per-image normalization、不做 threshold、不轉 uint8。
# ============================================================

@torch.no_grad()
def export_private_anomaly_maps(
    args,
    model,
    dataloader,
    device,
    split_name,
):
    if split_name not in {
        "test_private",
        "test_private_mixed",
    }:
        raise ValueError(
            f"Unsupported private split: {split_name}"
        )

    model.eval()

    output_root = os.path.join(
        args.output_dir,
        "anomaly_images",
        args.item,
        split_name,
    )

    os.makedirs(
        output_root,
        exist_ok=True,
    )

    gaussian_kernel = get_gaussian_kernel(
        kernel_size=5,
        sigma=4,
    ).to(device)

    saved_count = 0

    for img, img_paths in tqdm(
        dataloader,
        ncols=90,
        desc=f"Export {args.item}/{split_name}",
    ):
        img = img.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # 原 INP-Former inference / anomaly-map pipeline
        # ----------------------------------------------------
        output = model(img)
        en, de = output[0], output[1]

        anomaly_map, _ = cal_anomaly_maps(
            en,
            de,
            img.shape[-1],
        )

        if args.resize_mask is not None:
            anomaly_map = F.interpolate(
                anomaly_map,
                size=args.resize_mask,
                mode="bilinear",
                align_corners=False,
            )

        anomaly_map = gaussian_kernel(anomaly_map)

        batch_size = img.shape[0]

        for b_idx in range(batch_size):
            img_path = img_paths[b_idx]

            pred_map = (
                anomaly_map[b_idx]
                .squeeze()
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            if pred_map.ndim != 2:
                raise RuntimeError(
                    f"Private anomaly map must be 2D, "
                    f"got shape={pred_map.shape}, image={img_path}"
                )

            # 官方 checker 要求 float16。
            pred_map = pred_map.astype(
                np.float16,
                copy=False,
            )

            if not np.isfinite(pred_map).all():
                raise RuntimeError(
                    f"Anomaly map contains NaN/Inf: {img_path}"
                )

            stem = Path(img_path).stem
            save_path = os.path.join(
                output_root,
                f"{stem}.tiff",
            )

            # 防止不同來源子資料夾出現同名檔，造成靜默覆寫。
            if os.path.exists(save_path):
                raise FileExistsError(
                    f"Duplicate output filename detected: {save_path}"
                )

            tiff.imwrite(
                save_path,
                pred_map,
                dtype=np.float16,
            )

            saved_count += 1

    print_fn(
        f"[{args.item}] {split_name}: "
        f"saved {saved_count} float16 TIFF files to "
        f"{output_root}"
    )

    return saved_count


# ============================================================
# [2026-09-07 修改結束] MVTec AD 2 private TIFF export
# ============================================================

# ============================================================
# [2026-09-07 修改開始] MVTec AD 2 thresholded private export
#
# Threshold baseline suggested by the official checker:
#   segmentation_threshold = mean(anomaly_scores_val)
#                            + 3 * std(anomaly_scores_val)
#
# MVTec AD 2 validation 只有正常影像，因此這裡統計該 category
# validation 所有 pixel anomaly scores。使用串流 sum/sum-of-squares，
# 避免把所有 validation anomaly maps 同時留在 RAM。
# ============================================================

@torch.no_grad()
def estimate_validation_segmentation_threshold(
    args,
    model,
    dataloader,
    device,
):
    model.eval()

    gaussian_kernel = get_gaussian_kernel(
        kernel_size=5,
        sigma=4,
    ).to(device)

    score_sum = 0.0
    score_sq_sum = 0.0
    score_count = 0
    image_count = 0

    for (
        img,
        _gt,
        _label,
        _img_paths,
    ) in tqdm(
        dataloader,
        ncols=90,
        desc=f"Threshold {args.item}/validation",
    ):
        img = img.to(
            device,
            non_blocking=True,
        )

        output = model(img)
        en, de = output[0], output[1]

        anomaly_map, _ = cal_anomaly_maps(
            en,
            de,
            img.shape[-1],
        )

        if args.resize_mask is not None:
            anomaly_map = F.interpolate(
                anomaly_map,
                size=args.resize_mask,
                mode="bilinear",
                align_corners=False,
            )

        anomaly_map = gaussian_kernel(anomaly_map)

        values = (
            anomaly_map
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )

        if not np.isfinite(values).all():
            raise RuntimeError(
                f"Validation anomaly map contains NaN/Inf: {args.item}"
            )

        score_sum += float(values.sum(dtype=np.float64))
        score_sq_sum += float(
            np.square(values, dtype=np.float64).sum(dtype=np.float64)
        )
        score_count += int(values.size)
        image_count += int(img.shape[0])

    if score_count == 0:
        raise RuntimeError(
            f"No validation anomaly scores found for {args.item}"
        )

    mean = score_sum / score_count
    variance = max(
        0.0,
        score_sq_sum / score_count - mean * mean,
    )
    std = float(np.sqrt(variance))

    threshold = float(
        mean + args.threshold_std_factor * std
    )

    threshold_dir = os.path.join(
        args.output_dir,
        "thresholds",
    )
    os.makedirs(
        threshold_dir,
        exist_ok=True,
    )

    threshold_path = os.path.join(
        threshold_dir,
        f"{args.item}.txt",
    )

    with open(
        threshold_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(f"category={args.item}\n")
        f.write(f"validation_images={image_count}\n")
        f.write(f"pixel_count={score_count}\n")
        f.write(f"mean={mean:.12g}\n")
        f.write(f"std={std:.12g}\n")
        f.write(
            f"std_factor={args.threshold_std_factor:.12g}\n"
        )
        f.write(f"threshold={threshold:.12g}\n")

    print_fn(
        f"[{args.item}] validation threshold: "
        f"mean={mean:.8f}, std={std:.8f}, "
        f"threshold={threshold:.8f} "
        f"(mean + {args.threshold_std_factor:g} * std)"
    )

    return threshold


@torch.no_grad()
def export_private_thresholded_anomaly_maps(
    args,
    model,
    dataloader,
    device,
    split_name,
    segmentation_threshold,
):
    if split_name not in {
        "test_private",
        "test_private_mixed",
    }:
        raise ValueError(
            f"Unsupported private split: {split_name}"
        )

    model.eval()

    output_root = os.path.join(
        args.output_dir,
        "anomaly_images_thresholded",
        args.item,
        split_name,
    )

    os.makedirs(
        output_root,
        exist_ok=True,
    )

    gaussian_kernel = get_gaussian_kernel(
        kernel_size=5,
        sigma=4,
    ).to(device)

    saved_count = 0

    for img, img_paths in tqdm(
        dataloader,
        ncols=90,
        desc=f"Export thresholded {args.item}/{split_name}",
    ):
        img = img.to(
            device,
            non_blocking=True,
        )

        output = model(img)
        en, de = output[0], output[1]

        anomaly_map, _ = cal_anomaly_maps(
            en,
            de,
            img.shape[-1],
        )

        if args.resize_mask is not None:
            anomaly_map = F.interpolate(
                anomaly_map,
                size=args.resize_mask,
                mode="bilinear",
                align_corners=False,
            )

        anomaly_map = gaussian_kernel(anomaly_map)

        for b_idx in range(img.shape[0]):
            img_path = img_paths[b_idx]

            pred_map = (
                anomaly_map[b_idx]
                .squeeze()
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            if pred_map.ndim != 2:
                raise RuntimeError(
                    f"Thresholded anomaly map must be 2D, "
                    f"got shape={pred_map.shape}, image={img_path}"
                )

            if not np.isfinite(pred_map).all():
                raise RuntimeError(
                    f"Anomaly map contains NaN/Inf: {img_path}"
                )

            # 官方要求 binary PNG：normal=0, anomaly=255。
            binary_map = (
                pred_map > segmentation_threshold
            ).astype(np.uint8) * 255

            stem = Path(img_path).stem
            save_path = os.path.join(
                output_root,
                f"{stem}.png",
            )

            if os.path.exists(save_path):
                raise FileExistsError(
                    f"Duplicate thresholded output filename detected: "
                    f"{save_path}"
                )

            Image.fromarray(
                binary_map,
                mode="L",
            ).save(save_path)

            saved_count += 1

    print_fn(
        f"[{args.item}] {split_name}: "
        f"saved {saved_count} thresholded PNG files to "
        f"{output_root}; threshold={segmentation_threshold:.8f}"
    )

    return saved_count


# ============================================================
# [2026-09-07 修改結束] MVTec AD 2 thresholded private export
# ============================================================


# ============================================================
# Run one category
# ============================================================

def run_one_category(
    args
):
    setup_seed(
        args.seed
    )

    (
        data_transform,
        gt_transform,
    ) = build_transforms(
        args
    )

    model, trainable = build_model(
        args
    )

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # Train / Test
    # --------------------------------------------------------
    if args.phase == "train":
        train_data, train_loader = (
            build_train_loader(
                args,
                data_transform,
            )
        )

        train_one_category(
            args,
            model,
            trainable,
            train_data,
            train_loader,
            device,
        )

    elif args.phase == "test":
        load_checkpoint(
            args,
            model,
            device,
        )

    else:
        raise ValueError(
            f"Unsupported phase: "
            f"{args.phase}"
        )

    # ========================================================
    # [2026-09-07 修改開始] selectable public/private inference
    #
    # test_public         : 保留既有 public TIFF export
    # test_private        : 官方 private regular
    # test_private_mixed  : 官方 private mixed
    # private_all         : 一次輸出兩個 private split
    # ========================================================

    saved_test = 0
    saved_private = 0
    saved_private_mixed = 0

    # ========================================================
    # [2026-09-07 修改開始] thresholded private preparation
    # 若要求輸出 thresholded PNG，先用該 category 的 validation
    # normal anomaly scores 建立 segmentation threshold。
    # ========================================================
    segmentation_threshold = None
    saved_private_thresholded = 0
    saved_private_mixed_thresholded = 0

    if (
        args.export_thresholded_private
        and args.test_split in {
            "test_private",
            "test_private_mixed",
            "private_all",
        }
    ):
        threshold_val_data, threshold_val_loader = (
            build_validation_loader(
                args,
                data_transform,
                gt_transform,
            )
        )

        print_fn(
            f"[{args.item}] validation images for threshold: "
            f"{len(threshold_val_data)}"
        )

        segmentation_threshold = (
            estimate_validation_segmentation_threshold(
                args,
                model,
                threshold_val_loader,
                device,
            )
        )
    # ========================================================
    # [2026-09-07 修改結束] thresholded private preparation
    # ========================================================

    if args.test_split == "test_public":
        test_data, test_loader = (
            build_test_public_loader(
                args,
                data_transform,
                gt_transform,
            )
        )

        print_fn(
            f"[{args.item}] TESTpub images: "
            f"{len(test_data)}"
        )

        saved_test = (
            export_original_inp_anomaly_maps(
                args,
                model,
                test_loader,
                device,
                split_name="test_public",
            )
        )

    else:
        if args.test_split in {
            "test_private",
            "private_all",
        }:
            private_data, private_loader = (
                build_private_loader(
                    args,
                    data_transform,
                    split_name="test_private",
                )
            )

            print_fn(
                f"[{args.item}] test_private images: "
                f"{len(private_data)}"
            )

            if not args.thresholded_only:
                saved_private = (
                    export_private_anomaly_maps(
                        args,
                        model,
                        private_loader,
                        device,
                        split_name="test_private",
                    )
                )

            # ====================================================
            # [2026-09-07 修改開始] test_private thresholded PNG
            # ====================================================
            if args.export_thresholded_private:
                saved_private_thresholded = (
                    export_private_thresholded_anomaly_maps(
                        args,
                        model,
                        private_loader,
                        device,
                        split_name="test_private",
                        segmentation_threshold=segmentation_threshold,
                    )
                )
            # ====================================================
            # [2026-09-07 修改結束] test_private thresholded PNG
            # ====================================================

        if args.test_split in {
            "test_private_mixed",
            "private_all",
        }:
            mixed_data, mixed_loader = (
                build_private_loader(
                    args,
                    data_transform,
                    split_name="test_private_mixed",
                )
            )

            print_fn(
                f"[{args.item}] test_private_mixed images: "
                f"{len(mixed_data)}"
            )

            if not args.thresholded_only:
                saved_private_mixed = (
                    export_private_anomaly_maps(
                        args,
                        model,
                        mixed_loader,
                        device,
                        split_name="test_private_mixed",
                    )
                )

            # ====================================================
            # [2026-09-07 修改開始] test_private_mixed thresholded PNG
            # ====================================================
            if args.export_thresholded_private:
                saved_private_mixed_thresholded = (
                    export_private_thresholded_anomaly_maps(
                        args,
                        model,
                        mixed_loader,
                        device,
                        split_name="test_private_mixed",
                        segmentation_threshold=segmentation_threshold,
                    )
                )
            # ====================================================
            # [2026-09-07 修改結束] test_private_mixed thresholded PNG
            # ====================================================

    # ========================================================
    # [2026-09-07 修改結束] selectable public/private inference
    # ========================================================

    # --------------------------------------------------------
    # Validation TIFF (optional)
    # --------------------------------------------------------
    saved_val = 0

    if args.export_validation:
        val_data, val_loader = (
            build_validation_loader(
                args,
                data_transform,
                gt_transform,
            )
        )

        print_fn(
            f"[{args.item}] validation images: "
            f"{len(val_data)}"
        )

        saved_val = (
            export_original_inp_anomaly_maps(
                args,
                model,
                val_loader,
                device,
                split_name="validation",
            )
        )

    return {
        "category": args.item,
        "test_public": saved_test,
        "test_private": saved_private,
        "test_private_mixed": saved_private_mixed,
        "test_private_thresholded": saved_private_thresholded,
        "test_private_mixed_thresholded": saved_private_mixed_thresholded,
        "validation": saved_val,
        "segmentation_threshold": segmentation_threshold,
    }


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "INP-Former single-class MVTec AD 2 "
            "train/test + original-pipeline TIFF export"
        )
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help=(
            "MVTec AD 2 root，例如 "
            "/home/.../mvtec_ad_2"
        ),
    )

    parser.add_argument(
        "--phase",
        type=str,
        default="test",
        choices=[
            "train",
            "test",
        ],
    )

    parser.add_argument(
        "--item",
        type=str,
        default=None,
        choices=MVTEC_AD2_CATEGORIES,
        help=(
            "指定單一 category；"
            "不指定則依序處理全部 8 類。"
        ),
    )

    # checkpoint
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./saved_results",
    )

    parser.add_argument(
        "--save_name",
        type=str,
        default=(
            "INP-Former-Single-Class-MVTec-AD-2"
        ),
    )

    # TIFF output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./anomaly_map_results",
    )

    # ========================================================
    # [2026-09-07 修改開始] 選擇要輸出的測試 split
    # ========================================================
    parser.add_argument(
        "--test_split",
        type=str,
        default="test_public",
        choices=[
            "test_public",
            "test_private",
            "test_private_mixed",
            "private_all",
        ],
        help=(
            "test_public: 輸出公開測試集；"
            "test_private: 輸出官方 private regular；"
            "test_private_mixed: 輸出官方 private mixed；"
            "private_all: 同時輸出兩個 private split。"
        ),
    )
    # ========================================================
    # [2026-09-07 修改結束] 選擇要輸出的測試 split
    # ========================================================

    # model
    parser.add_argument(
        "--encoder",
        type=str,
        default="dinov2reg_vit_base_14",
    )

    parser.add_argument(
        "--input_size",
        type=int,
        default=252,
    )

    parser.add_argument(
        "--crop_size",
        type=int,
        default=252,
    )

    parser.add_argument(
        "--INP_num",
        type=int,
        default=6,
    )

    # 重要：
    # 這就是原 evaluation_batch 的 resize_mask。
    # 如果你原本是 evaluation_batch(... resize_mask=256)
    # 就維持預設 256。
    parser.add_argument(
        "--resize_mask",
        type=int,
        default=256,
        help=(
            "完全對應原 evaluation_batch 的 resize_mask。"
            "預設 256，因此 TIFF 會是 256x256。"
        ),
    )

    # training
    parser.add_argument(
        "--total_epochs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    # image score debug CSV only
    parser.add_argument(
        "--max_ratio",
        type=float,
        default=0.01,
    )

    # ========================================================
    # [2026-09-07 修改開始] thresholded private submission args
    # ========================================================
    parser.add_argument(
        "--export_thresholded_private",
        action="store_true",
        help=(
            "同時輸出官方 threshold-dependent metrics 所需的 "
            "anomaly_images_thresholded PNG。threshold 由該 category "
            "validation 正常像素分數 mean + k*std 自動估計。"
        ),
    )

    parser.add_argument(
        "--thresholded_only",
        action="store_true",
        help=(
            "只補 anomaly_images_thresholded PNG，不重新輸出 TIFF。"
            "適合已經有 checker 通過的 anomaly_images submission。"
        ),
    )

    parser.add_argument(
        "--threshold_std_factor",
        type=float,
        default=3.0,
        help=(
            "validation threshold 的 std 倍數；"
            "官方 checker 提示的 baseline 為 3.0。"
        ),
    )
    # ========================================================
    # [2026-09-07 修改結束] thresholded private submission args
    # ========================================================

    parser.add_argument(
        "--export_validation",
        action="store_true",
        help=(
            "同時輸出 validation TIFF，"
            "方便統一算分程式算 MVTec AD 2 F1 threshold。"
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    # ========================================================
    # [2026-09-07 修改開始] thresholded-only argument checks
    # ========================================================
    if args.thresholded_only:
        args.export_thresholded_private = True

        if args.test_split == "test_public":
            raise ValueError(
                "--thresholded_only 僅適用於 test_private / "
                "test_private_mixed / private_all。"
            )

        if args.phase != "test":
            raise ValueError(
                "--thresholded_only 必須搭配 --phase test。"
            )
    # ========================================================
    # [2026-09-07 修改結束] thresholded-only argument checks
    # ========================================================

    # ========================================================
    # [2026-09-07 修改開始] private submission mode 提示
    # ========================================================
    if args.test_split in {
        "test_private",
        "test_private_mixed",
        "private_all",
    }:
        print(
            "[2026-09-07] MVTec AD 2 private submission mode: "
            "TIFF 將輸出為 float16 continuous anomaly maps。"
        )
    # ========================================================
    # [2026-09-07 修改結束] private submission mode 提示
    # ========================================================

    # MVTec AD 2 INP-Former sanity checks
    if args.INP_num != 6:
        raise ValueError(
            "MVTec AD 2 的 INP-Former baseline "
            "應使用 6 prototypes。"
        )

    if (
        "base" not in args.encoder
        or "_14" not in args.encoder
    ):
        raise ValueError(
            "MVTec AD 2 的 INP-Former baseline "
            "應使用 DINOv2 ViT-B/14。"
        )

    if args.input_size % 14 != 0:
        raise ValueError(
            "input_size 必須是 14 的倍數。"
        )

    if (
        args.crop_size
        != args.input_size
    ):
        raise ValueError(
            "MVTec AD 2 版本請不要 CenterCrop；"
            "crop_size 應等於 input_size。"
        )

    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    # 注意：
    # checkpoint 路徑名稱要和你 train 時一致。
    run_name = (
        args.save_name
        + f"_Encoder={args.encoder}"
        + f"_Resize={args.input_size}"
        + f"_INP_num={args.INP_num}"
    )

    args.save_name = (
        run_name
    )

    logger = get_logger(
        run_name,
        os.path.join(
            args.save_dir,
            run_name,
        ),
    )

    print_fn = logger.info

    if args.item is None:
        item_list = (
            MVTEC_AD2_CATEGORIES
        )
    else:
        item_list = [
            args.item
        ]

    summary = []

    for item in item_list:
        args.item = item

        result = run_one_category(
            args
        )

        summary.append(
            result
        )

    print_fn(
        "=" * 80
    )

    print_fn(
        "TIFF export summary"
    )

    # ========================================================
    # [2026-09-07 修改開始] summary 加入 private split 計數
    # ========================================================
    for result in summary:
        print_fn(
            (
                "{}: TESTpub={}, test_private={}, "
                "test_private_mixed={}, private_thr={}, "
                "mixed_thr={}, validation={}, threshold={}"
            ).format(
                result["category"],
                result["test_public"],
                result["test_private"],
                result["test_private_mixed"],
                result["test_private_thresholded"],
                result["test_private_mixed_thresholded"],
                result["validation"],
                (
                    "None"
                    if result["segmentation_threshold"] is None
                    else f'{result["segmentation_threshold"]:.8f}'
                ),
            )
        )
    # ========================================================
    # [2026-09-07 修改結束] summary 加入 private split 計數
    # ========================================================

    print_fn(
        f"Output root: "
        f"{os.path.abspath(args.output_dir)}"
    )
