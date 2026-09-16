import os
import glob
import json
import warnings

import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


torch.multiprocessing.set_sharing_strategy('file_system')


# ============================================================
# MVTec AD 2 / INP-Former preprocessing
# ============================================================

def get_data_transforms(
        size=252,
        isize=None,          # 保留此參數是為了相容舊主程式，但不再 CenterCrop
        mean_train=None,
        std_train=None
    ):

    """
    MVTec AD 2 benchmark-oriented preprocessing for INP-Former.

    官方論文：
    - INP-Former 使用 ViT-B DINOv2
    - patch size = 14
    - input size 調整到 patch size 的倍數
    - resolution 保持接近 256 x 256

    因此預設使用 252 x 252 = 18 * 14。

    注意：
    - 不使用 CenterCrop，避免裁掉 MVTec AD 2 的 border defects。
    - RGB image 使用 bilinear resize。
    - Ground-truth mask 使用 nearest-neighbor resize。
    """

    if size % 14 != 0:
        raise ValueError(
            f"For INP-Former / DINOv2 patch size 14, "
            f"size should be divisible by 14, but got size={size}."
        )

    # 舊程式可能仍會傳入 isize=224 / 256。
    # MVTec AD 2 baseline 不應再進行 center crop。
    if isize is not None and isize != size:
        warnings.warn(
            f"isize={isize} is ignored. "
            f"CenterCrop is disabled for MVTec AD 2. "
            f"Final input size is {size}x{size}."
        )

    mean_train = (
        [0.485, 0.456, 0.406]
        if mean_train is None
        else mean_train
    )

    std_train = (
        [0.229, 0.224, 0.225]
        if std_train is None
        else std_train
    )

    # --------------------------------------------------------
    # RGB image transform
    # --------------------------------------------------------
    data_transforms = transforms.Compose([
        transforms.Resize(
            (size, size),
            interpolation=InterpolationMode.BILINEAR
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=mean_train,
            std=std_train
        )
    ])

    # --------------------------------------------------------
    # Segmentation mask transform
    # --------------------------------------------------------
    gt_transforms = transforms.Compose([
        transforms.Resize(
            (size, size),
            interpolation=InterpolationMode.NEAREST
        ),

        transforms.ToTensor()
    ])

    return data_transforms, gt_transforms


# ============================================================
# MVTec AD 2 Dataset
# ============================================================

class MVTecDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root,
        transform,
        gt_transform,
        phase
    ):
        """
        phase:
            'train'
            'validation' / 'val'
            'test_public' / 'test'

        Expected structure:

        category/
        ├── train/
        │   └── good/
        │
        ├── validation/
        │   └── good/
        │
        └── test_public/
            ├── good/
            ├── <defect_type>/
            └── ground_truth/
                └── <defect_type>/
        """

        self.root = root
        self.transform = transform
        self.gt_transform = gt_transform

        phase = phase.lower()

        # ----------------------------------------------------
        # Normalize phase names
        # ----------------------------------------------------
        if phase == 'train':
            self.phase = 'train'

        elif phase in ['val', 'validation']:
            self.phase = 'validation'

        elif phase in ['test', 'test_public']:
            self.phase = 'test_public'

        else:
            raise ValueError(
                f"Unsupported phase '{phase}'. "
                f"Use train, validation, or test_public."
            )

        # ----------------------------------------------------
        # Dataset paths
        # ----------------------------------------------------
        if self.phase == 'train':

            self.img_path = os.path.join(
                root,
                'train'
            )

            self.gt_path = None

        elif self.phase == 'validation':

            self.img_path = os.path.join(
                root,
                'validation'
            )

            self.gt_path = None

        else:

            self.img_path = os.path.join(
                root,
                'test_public'
            )

            self.gt_path = os.path.join(
                self.img_path,
                'ground_truth'
            )

        if not os.path.isdir(self.img_path):
            raise FileNotFoundError(
                f"Dataset directory does not exist: "
                f"{self.img_path}"
            )

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------
        (
            self.img_paths,
            self.gt_paths,
            self.labels,
            self.types
        ) = self.load_dataset()

        self.cls_idx = 0

        print(
            f"[MVTec AD 2] "
            f"phase={self.phase}, "
            f"samples={len(self.img_paths)}"
        )


    # ========================================================
    # Utility: collect images
    # ========================================================

    @staticmethod
    def _find_images(folder):

        if not os.path.isdir(folder):
            return []

        extensions = [
            '*.png',
            '*.PNG',
            '*.jpg',
            '*.JPG',
            '*.jpeg',
            '*.JPEG',
            '*.bmp',
            '*.BMP',
            '*.tif',
            '*.tiff'
        ]

        paths = []

        for ext in extensions:
            paths.extend(
                glob.glob(
                    os.path.join(folder, ext)
                )
            )

        return sorted(paths)


    # ========================================================
    # Utility: match image and GT by filename
    # ========================================================

    def _find_gt_for_image(
        self,
        img_path,
        gt_folder
    ):

        """
        不依賴 img_paths.sort() / gt_paths.sort() 的位置配對。

        會依照 image stem 尋找：
            xxx.png
            xxx_mask.png

        如果你的 MVTec AD 2 GT 命名不同，
        只需要修改這個函式。
        """

        stem = os.path.splitext(
            os.path.basename(img_path)
        )[0]

        extensions = [
            '.png',
            '.PNG',
            '.jpg',
            '.JPG',
            '.bmp',
            '.tif',
            '.tiff'
        ]

        possible_stems = [
            stem,
            stem + '_mask'
        ]

        candidates = []

        for gt_stem in possible_stems:

            for ext in extensions:

                candidate = os.path.join(
                    gt_folder,
                    gt_stem + ext
                )

                if os.path.isfile(candidate):
                    candidates.append(candidate)

        if len(candidates) == 0:
            raise FileNotFoundError(
                f"\nCannot find ground truth.\n"
                f"Image : {img_path}\n"
                f"GT dir: {gt_folder}\n"
            )

        # 若同一張 image 對到多個 mask，
        # 直接報錯，不偷偷選其中之一。
        if len(candidates) > 1:
            raise RuntimeError(
                f"Multiple GT files found for:\n"
                f"{img_path}\n"
                f"{candidates}"
            )

        return candidates[0]


    # ========================================================
    # Load dataset
    # ========================================================

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        # ====================================================
        # TRAIN / VALIDATION
        #
        # MVTec AD 2:
        # train 和 validation 都只有正常資料
        # ====================================================

        if self.phase in [
            'train',
            'validation'
        ]:

            good_path = os.path.join(
                self.img_path,
                'good'
            )

            # 某些資料整理方式可能直接把圖片
            # 放在 train/ 或 validation/ 底下。
            if os.path.isdir(good_path):

                img_paths = self._find_images(
                    good_path
                )

            else:

                img_paths = self._find_images(
                    self.img_path
                )

            if len(img_paths) == 0:
                raise RuntimeError(
                    f"No images found in "
                    f"{self.img_path}"
                )

            for img_path in img_paths:

                img_tot_paths.append(
                    img_path
                )

                # normal image 沒有 GT file
                gt_tot_paths.append(
                    None
                )

                # normal
                tot_labels.append(
                    0
                )

                tot_types.append(
                    'good'
                )

        # ====================================================
        # TESTpub
        # ====================================================

        elif self.phase == 'test_public':

            defect_types = sorted(
                os.listdir(
                    self.img_path
                )
            )

            for defect_type in defect_types:

                # Ground-truth directory 本身不是 class
                if defect_type == 'ground_truth':
                    continue

                defect_folder = os.path.join(
                    self.img_path,
                    defect_type
                )

                if not os.path.isdir(
                    defect_folder
                ):
                    continue

                img_paths = self._find_images(
                    defect_folder
                )

                # --------------------------------------------
                # Normal
                # --------------------------------------------
                if defect_type == 'good':

                    for img_path in img_paths:

                        img_tot_paths.append(
                            img_path
                        )

                        gt_tot_paths.append(
                            None
                        )

                        tot_labels.append(
                            0
                        )

                        tot_types.append(
                            'good'
                        )

                # --------------------------------------------
                # Anomaly
                # --------------------------------------------
                else:

                    gt_folder = os.path.join(
                        self.gt_path,
                        defect_type
                    )

                    if not os.path.isdir(
                        gt_folder
                    ):
                        raise FileNotFoundError(
                            f"GT directory does not exist: "
                            f"{gt_folder}"
                        )

                    for img_path in img_paths:

                        gt_path = self._find_gt_for_image(
                            img_path,
                            gt_folder
                        )

                        img_tot_paths.append(
                            img_path
                        )

                        gt_tot_paths.append(
                            gt_path
                        )

                        tot_labels.append(
                            1
                        )

                        tot_types.append(
                            defect_type
                        )

        # ====================================================
        # Sanity checks
        # ====================================================

        if len(img_tot_paths) == 0:
            raise RuntimeError(
                f"No images loaded from "
                f"{self.img_path}"
            )

        assert (
            len(img_tot_paths)
            == len(gt_tot_paths)
            == len(tot_labels)
            == len(tot_types)
        ), (
            "Dataset length mismatch:\n"
            f"images = {len(img_tot_paths)}\n"
            f"GT     = {len(gt_tot_paths)}\n"
            f"labels = {len(tot_labels)}\n"
            f"types  = {len(tot_types)}"
        )

        return (
            np.array(
                img_tot_paths,
                dtype=object
            ),

            np.array(
                gt_tot_paths,
                dtype=object
            ),

            np.array(
                tot_labels,
                dtype=np.int64
            ),

            np.array(
                tot_types,
                dtype=object
            )
        )


    def __len__(self):

        return len(
            self.img_paths
        )


    # ========================================================
    # Get sample
    # ========================================================

    def __getitem__(self, idx):

        img_path = self.img_paths[idx]

        gt_path = self.gt_paths[idx]

        label = int(
            self.labels[idx]
        )

        img_type = self.types[idx]

        # ----------------------------------------------------
        # RGB image
        # ----------------------------------------------------

        img = Image.open(
            img_path
        ).convert('RGB')

        img = self.transform(
            img
        )

        H = img.shape[-2]
        W = img.shape[-1]

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        if label == 0:

            # 原本程式是：
            #
            # [1, H, H]
            #
            # 這裡修正成：
            #
            # [1, H, W]

            gt = torch.zeros(
                (1, H, W),
                dtype=torch.float32
            )

        else:

            gt = Image.open(
                gt_path
            ).convert('L')

            gt = self.gt_transform(
                gt
            )

            # 強制 binary
            #
            # 避免 interpolation /
            # image format 導致非 0/1 值。
            gt = (
                gt > 0.5
            ).float()

        # ----------------------------------------------------
        # Sanity check
        # ----------------------------------------------------

        assert (
            img.shape[-2:]
            == gt.shape[-2:]
        ), (
            f"\nImage / GT size mismatch\n"
            f"image : {img.shape}\n"
            f"GT    : {gt.shape}\n"
            f"path  : {img_path}"
        )

        return (
            img,
            gt,
            label,
            img_path
        )