import os
from typing import List, Tuple, Optional
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dataloaders.data_processing import get_mask_from_json

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".npy")
essential_layers = list(range(0, 41))


class SimiLoss:
    """A class to compute various losses for similarity maps against ground truth masks."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the LossComputer.
        
        Args:
            device: PyTorch device to use for computations. If None, will auto-detect.
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _prepare_pos_weight(self, pos_weight, dtype, device):
        """Helper function to prepare pos_weight tensor safely."""
        if pos_weight is None:
            return None
        elif isinstance(pos_weight, torch.Tensor):
            return pos_weight.clone().detach().to(dtype=dtype, device=device)
        else:
            return torch.tensor(pos_weight, dtype=dtype, device=device)
        
    def sigmoid_ce_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        """Compute sigmoid cross-entropy loss with optional positive class weighting."""
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none", pos_weight=pos_weight)
        loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
        return loss

    def dice_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        eps: float = 1e-6,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        """Compute Dice loss from logits with optional positive class weighting."""
        probs = inputs.sigmoid()
        probs = probs.flatten(1, 2)
        targets = targets.flatten(1, 2)
        
        if pos_weight is not None:
            # Apply pos_weight to targets for weighted Dice
            weighted_targets = targets * pos_weight
            numerator = 2 * (probs * weighted_targets).sum(-1)
            denominator = probs.sum(-1) + weighted_targets.sum(-1)
        else:
            numerator = 2 * (probs * targets).sum(-1)
            denominator = probs.sum(-1) + targets.sum(-1)
        
        loss = 1 - (numerator + eps) / (denominator + eps)
        loss = loss.flatten(0).sum() / (num_masks + 1e-8)
        return loss

    def compute_dice_loss(self, pred_logits: torch.Tensor, gt_mask: torch.Tensor, num_masks: float, pos_weight=None) -> torch.Tensor:
        """Compute Dice loss with auto pos_weight computation."""
        # Ensure tensors are on the same device
        pred_logits = pred_logits.to(self.device)
        gt_mask = gt_mask.to(self.device)
        
        # Reshape to (B, H, W) format (no need to add channel dim)
        B, H, W = pred_logits.shape
        # No reshaping needed - keep as (B, H, W)
        
        # Auto-compute pos_weight if not provided
        pos_weight_tensor = None
        if pos_weight is None:
            # Compute neg/pos ratio from ground truth
            eps = 1e-6
            num_pos = float(gt_mask.sum().item())
            num_total = float(gt_mask.numel())
            num_neg = max(num_total - num_pos, 0.0)
            auto_pos_weight = num_neg / max(num_pos, eps)
            pos_weight_tensor = torch.tensor(auto_pos_weight, dtype=pred_logits.dtype, device=self.device)
        elif pos_weight is not None:
            pos_weight_tensor = self._prepare_pos_weight(pos_weight, pred_logits.dtype, self.device)
        
        # Compute Dice loss directly on (B, H, W) format
        probs = pred_logits.sigmoid()
        probs = probs.flatten(1)  # (B, H*W)
        targets = gt_mask.flatten(1)  # (B, H*W)
        
        if pos_weight_tensor is not None:
            # Apply pos_weight to targets for weighted Dice
            weighted_targets = targets * pos_weight_tensor
            numerator = 2 * (probs * weighted_targets).sum(-1)
            denominator = probs.sum(-1) + weighted_targets.sum(-1)
        else:
            numerator = 2 * (probs * targets).sum(-1)
            denominator = probs.sum(-1) + targets.sum(-1)
        
        loss = 1 - (numerator + 1e-6) / (denominator + 1e-6)
        loss = loss.sum() / (num_masks + 1e-8)
        return loss

    def compute_bce_loss(self, pred_logits: torch.Tensor, gt_mask: torch.Tensor, num_masks: float, pos_weight=None) -> torch.Tensor:
        """Compute BCE loss with auto pos_weight computation."""
        # Ensure tensors are on the same device
        pred_logits = pred_logits.to(self.device)
        gt_mask = gt_mask.to(self.device)
        
        # Keep as (B, H, W) format (no need to add channel dim)
        B, H, W = pred_logits.shape
        # No reshaping needed - keep as (B, H, W)
        
        # Auto-compute pos_weight if not provided
        pos_weight_tensor = None
        if pos_weight is None:
            # Compute neg/pos ratio from ground truth
            eps = 1e-6
            num_pos = float(gt_mask.sum().item())
            num_total = float(gt_mask.numel())
            num_neg = max(num_total - num_pos, 0.0)
            auto_pos_weight = num_neg / max(num_pos, eps)
            pos_weight_tensor = torch.tensor(auto_pos_weight, dtype=pred_logits.dtype, device=self.device)
        elif pos_weight is not None:
            pos_weight_tensor = self._prepare_pos_weight(pos_weight, pred_logits.dtype, self.device)
        
        # Compute BCE loss directly on (B, H, W) format
        loss = F.binary_cross_entropy_with_logits(pred_logits, gt_mask, reduction="none", pos_weight=pos_weight_tensor)
        loss = loss.flatten(1).mean(1).sum() / (num_masks + 1e-8)
        return loss

    def compute_gaussian_bce_loss(self, pred_logits: torch.Tensor, gt_mask: torch.Tensor, num_masks: float, pos_weight=None, 
                                 ksize: int = 31, sigma: float = 7.0) -> torch.Tensor:
        """Compute BCE loss against Gaussian soft labels."""
        # Ensure tensors are on the same device
        pred_logits = pred_logits.to(self.device)
        gt_mask = gt_mask.to(self.device)
        
        # Keep as (B, H, W) format (no need to add channel dim)
        B, H, W = pred_logits.shape
        # No reshaping needed - keep as (B, H, W)
        
        # Generate Gaussian heatmap for batch of masks (batch processing)
        gaussian_targets = self.gaussian_heatmap_from_mask(gt_mask, ksize=ksize, sigma=sigma, normalize=True)
        # gaussian_targets is already (B, H, W) format
        
        # Auto-compute pos_weight if not provided
        pos_weight_tensor = None
        if pos_weight is None:
            # Compute ratio using soft counts from Gaussian heatmap
            eps = 1e-6
            sum_pos_soft = float(gaussian_targets.sum().item())
            sum_neg_soft = float(gaussian_targets.numel()) - sum_pos_soft
            auto_pos_weight = sum_neg_soft / max(sum_pos_soft, eps)
            pos_weight_tensor = torch.tensor(auto_pos_weight, dtype=pred_logits.dtype, device=self.device)
        elif pos_weight is not None:
            pos_weight_tensor = self._prepare_pos_weight(pos_weight, pred_logits.dtype, self.device)
        # Compute BCE loss directly on (B, H, W) format
        loss = F.binary_cross_entropy_with_logits(pred_logits, gaussian_targets, reduction="none", pos_weight=pos_weight_tensor)
        loss = loss.flatten(1).mean(1).sum() / (num_masks + 1e-8)
        return loss

    def compute_gaussian_dice_loss(self, pred_logits: torch.Tensor, gt_mask: torch.Tensor, num_masks: float, pos_weight=None,
                                  ksize: int = 31, sigma: float = 7.0) -> torch.Tensor:
        """Compute Dice loss against Gaussian soft labels."""
        # Ensure tensors are on the same device
        pred_logits = pred_logits.to(self.device)
        gt_mask = gt_mask.to(self.device)
        
        # Keep as (B, H, W) format (no need to add channel dim)
        B, H, W = pred_logits.shape
        # No reshaping needed - keep as (B, H, W)
        
        # Generate Gaussian heatmap for batch of masks (batch processing)
        gaussian_targets = self.gaussian_heatmap_from_mask(gt_mask, ksize=ksize, sigma=sigma, normalize=True)
        # gaussian_targets is already (B, H, W) format
        
        # Auto-compute pos_weight if not provided
        pos_weight_tensor = None
        if pos_weight is None:
            # Compute ratio using soft counts from Gaussian heatmap
            eps = 1e-6
            sum_pos_soft = float(gaussian_targets.sum().item())
            sum_neg_soft = float(gaussian_targets.numel()) - sum_pos_soft
            auto_pos_weight = sum_neg_soft / max(sum_pos_soft, eps)
            pos_weight_tensor = torch.tensor(auto_pos_weight, dtype=pred_logits.dtype, device=self.device)
        elif pos_weight is not None:
            pos_weight_tensor = self._prepare_pos_weight(pos_weight, pred_logits.dtype, self.device)
        
        # Compute Dice loss directly on (B, H, W) format
        probs = pred_logits.sigmoid()
        probs = probs.flatten(1)  # (B, H*W)
        targets = gaussian_targets.flatten(1)  # (B, H*W)
        
        if pos_weight_tensor is not None:
            # Apply pos_weight to targets for weighted Dice
            weighted_targets = targets * pos_weight_tensor
            numerator = 2 * (probs * weighted_targets).sum(-1)
            denominator = probs.sum(-1) + weighted_targets.sum(-1)
        else:
            numerator = 2 * (probs * targets).sum(-1)
            denominator = probs.sum(-1) + targets.sum(-1)
        
        loss = 1 - (numerator + 1e-6) / (denominator + 1e-6)
        loss = loss.sum() / (num_masks + 1e-8)
        return loss


    def gaussian_heatmap_from_mask(self, binary_masks: torch.Tensor, ksize: int = 31, sigma: float = 7.0, normalize: bool = True) -> torch.Tensor:
        """Generate Gaussian heatmap for a batch of binary masks using PyTorch operations."""
        # Ensure odd kernel size and adapt to image size
        k = int(ksize)
        if k % 2 == 0:
            k += 1
        
        # Adapt kernel size to image dimensions
        min_dim = min(binary_masks.shape[1], binary_masks.shape[2])
        k = min(k, min_dim)
        if k % 2 == 0:
            k = max(3, k - 1)  # Ensure odd and at least 3
        
        # Create Gaussian kernel on the same device as input
        kernel_size = k
        device = binary_masks.device
        sigma_tensor = torch.tensor(sigma, dtype=torch.float32, device=device)
        
        # Create coordinate grid
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2
        x_coords, y_coords = torch.meshgrid(coords, coords, indexing='ij')
        
        # Compute Gaussian kernel
        kernel = torch.exp(-(x_coords**2 + y_coords**2) / (2 * sigma_tensor**2))
        kernel = kernel / kernel.sum()  # Normalize kernel
        
        # Apply convolution using F.conv2d
        mask_f = binary_masks.float().unsqueeze(1)  # Add channel dim (B, 1, H, W)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims (1, 1, k, k)
        
        # Pad the input with adaptive padding size
        pad_size = min(kernel_size // 2, min(mask_f.shape[2], mask_f.shape[3]) // 2)
        if pad_size > 0:
            mask_padded = F.pad(mask_f, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        else:
            mask_padded = mask_f
        
        # Apply convolution
        heat = F.conv2d(mask_padded, kernel, padding=0)  # (B, 1, H, W)
        heat = heat.squeeze(1)  # Remove channel dim (B, H, W)
        
        if normalize:
            # Normalize each sample in the batch individually
            if heat.shape[0] > 0:  # Check if batch is not empty
                vmax = heat.view(heat.shape[0], -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)  # (B, 1, 1)
                vmax = torch.clamp(vmax, min=1e-6)
                heat = heat / vmax
        
        return heat.float()

    def _load_similarity_map_probs_and_logits(self, path: str, target_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load similarity map and return continuous map and logits.
        
        Returns (cont_01, logits):
        - cont_01: a 0..1 continuous map derived from the file without assuming sigmoid
        - logits: logits if we can derive from probabilities; otherwise, fall back to raw array as logits.
        """
        h, w = target_size

        def _ensure_hw(arr: torch.Tensor) -> torch.Tensor:
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                # Convert BGR to grayscale using torch operations
                arr = arr[:, :, 2] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 0] * 0.114
            if arr.ndim == 3:
                arr = arr[..., 0]
            if arr.shape != (h, w):
                arr = F.interpolate(arr.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze()
            return arr.float()

        if path.lower().endswith(".npy"):
            arr = torch.from_numpy(np.load(path))
            arr = _ensure_hw(arr)
            # Treat .npy content as logits
            logits = arr.float()
            # Derive a 0..1 map via sigmoid for use in MSE against Gaussian heatmap
            cont = torch.sigmoid(logits)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Failed to read similarity map: {path}")
            img = torch.from_numpy(img).float()
            if img.ndim == 3:
                # Convert BGR to grayscale using torch operations
                img = img[:, :, 2] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 0] * 0.114
            # Image intensities → 0..1 continuous directly
            if img.max() > 1.0:
                cont = img / 255.0
            else:
                cont = torch.clamp(img, 0.0, 1.0)
            # Derive logits for other metrics that expect logits
            logits = torch.logit(cont, eps=1e-6)
            if cont.shape != (h, w):
                cont = F.interpolate(cont.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze()
                logits = F.interpolate(logits.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze()

        return cont.float(), logits.float()

    def _collect_ids_from_layer_dir(self, layer_dir: str) -> List[str]:
        """Collect all sample IDs from a layer directory."""
        ids: List[str] = []
        for f in os.listdir(layer_dir):
            path = os.path.join(layer_dir, f)
            if os.path.isfile(path) and f.lower().endswith(".npy"):
                ids.append(os.path.splitext(f)[0])
        ids.sort()
        return ids

    def _find_pred_for_id(self, layer_dir: str, sample_id: str) -> str:
        """Find prediction file for a given sample ID."""
        # Only accept .npy (logits) files
        target = os.path.join(layer_dir, f"{sample_id}.npy")
        if os.path.exists(target):
            return target
        raise FileNotFoundError(f"Prediction .npy not found for id {sample_id} in {layer_dir}")

    def compute_per_layer_losses_for_id(
        self,
        predictions_root: str,
        dataset_root: str,
        sample_id: str,
        viz_out_dir: str = "",
        pos_weight_user: float = -1.0,
        pos_weight_gauss_user: float = -1.0,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Compute losses for all layers for a single sample ID.
        
        Returns:
            Tuple of (bce_losses, bce_gauss_losses, dice_losses, dice_gauss_losses)
        """
        img_path = os.path.join(dataset_root, f"{sample_id}.jpg")
        if not os.path.exists(img_path):
            alt_img = os.path.join(dataset_root, f"{sample_id}.png")
            if os.path.exists(alt_img):
                img_path = alt_img
            else:
                raise FileNotFoundError(f"Image not found for id {sample_id}: {img_path}")

        json_path = os.path.join(dataset_root, f"{sample_id}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON not found for id {sample_id}: {json_path}")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        img_rgb = img_bgr[:, :, ::-1]

        mask, _, _ = get_mask_from_json(json_path, img_rgb)
        h, w = mask.shape

        valid_mask = (mask != 255)
        gt_bin = (mask == 1).astype(np.float32)

        valid_indices = np.where(valid_mask)
        if valid_indices[0].size == 0:
            raise ValueError(f"No valid pixels for id {sample_id}")

        # Convert to torch tensors
        gt_bin_tensor = torch.from_numpy(gt_bin).to(self.device)
        valid_mask_tensor = torch.from_numpy(valid_mask).to(self.device)

        # Build Gaussian heatmap from GT once per image (using batch function)
        gt_bin_batch = gt_bin_tensor.unsqueeze(0)  # Add batch dimension (1, H, W)
        gt_heat = self.gaussian_heatmap_from_mask(gt_bin_batch, ksize=31, sigma=7.0, normalize=True)
        gt_heat = gt_heat.squeeze(0)  # Remove batch dimension (H, W)
        
        # Select valid pixels
        gt_selected = gt_bin_tensor[valid_mask_tensor]
        heat_selected = gt_heat[valid_mask_tensor]

        gt_tensor = gt_selected.reshape(1, 1, -1)
        heat_tensor = heat_selected.reshape(1, 1, -1)

        # Compute pos_weight (auto if not provided)
        eps = 1e-6
        num_pos = float(gt_tensor.sum().item())
        num_total = float(gt_tensor.numel())
        num_neg = max(num_total - num_pos, 0.0)
        auto_pos_weight_val = num_neg / max(num_pos, eps)
        pos_weight_tensor = torch.tensor(
            auto_pos_weight_val if pos_weight_user <= 0 else pos_weight_user,
            dtype=gt_tensor.dtype,
            device=self.device,
        )

        # For Gaussian soft label, compute ratio using soft counts
        sum_pos_soft = float(heat_tensor.sum().item())
        sum_neg_soft = float(heat_tensor.numel()) - sum_pos_soft
        auto_pos_weight_gauss_val = sum_neg_soft / max(sum_pos_soft, eps)
        pos_weight_gauss_tensor = torch.tensor(
            auto_pos_weight_gauss_val if pos_weight_gauss_user <= 0 else pos_weight_gauss_user,
            dtype=gt_tensor.dtype,
            device=self.device,
        )

        bce_losses: List[float] = []
        bce_gauss_losses: List[float] = []
        dice_losses: List[float] = []
        dice_gauss_losses: List[float] = []

        # Prepare viz dir if requested
        per_id_viz_dir = ""
        if viz_out_dir:
            per_id_viz_dir = os.path.join(viz_out_dir, sample_id)
            os.makedirs(per_id_viz_dir, exist_ok=True)

        for layer in essential_layers:
            layer_dir = os.path.join(predictions_root, f"layer{layer}")
            pred_path = self._find_pred_for_id(layer_dir, sample_id)
            cont_map, logits_map = self._load_similarity_map_probs_and_logits(pred_path, (h, w))
            
            # Move to device
            cont_map = cont_map.to(self.device)
            logits_map = logits_map.to(self.device)

            logits_selected = logits_map[valid_mask_tensor]
            logits_tensor = logits_selected.reshape(1, 1, -1)

            # BCE with binary GT mask (with pos_weight)
            bce_val = self.compute_bce_loss(pred_logits=logits_tensor, gt_mask=gt_tensor, num_masks=1.0, pos_weight=pos_weight_tensor)
            # BCE with Gaussian soft label target (with pos_weight for soft positives)
            bce_gauss_val = self.compute_gaussian_bce_loss(pred_logits=logits_tensor, gt_mask=gt_tensor, num_masks=1.0, pos_weight=pos_weight_gauss_tensor)
            # Dice with binary GT mask (with pos_weight)
            dice_val = self.compute_dice_loss(pred_logits=logits_tensor, gt_mask=gt_tensor, num_masks=1.0, pos_weight=pos_weight_tensor)
            # Dice with Gaussian soft label target (with pos_weight for soft positives)
            dice_gauss_val = self.compute_gaussian_dice_loss(pred_logits=logits_tensor, gt_mask=gt_tensor, num_masks=1.0, pos_weight=pos_weight_gauss_tensor)

            bce_losses.append(float(bce_val.item()))
            bce_gauss_losses.append(float(bce_gauss_val.item()))
            dice_losses.append(float(dice_val.item()))
            dice_gauss_losses.append(float(dice_gauss_val.item()))

            # Visualization (convert back to numpy for cv2 operations)
            if per_id_viz_dir:
                cont_map_np = cont_map.cpu().numpy()
                gt_heat_np = gt_heat.cpu().numpy()
                self._save_layer_visualizations(
                    out_dir=per_id_viz_dir,
                    sample_id=sample_id,
                    layer=layer,
                    img_bgr=img_bgr,
                    cont_map=cont_map_np,
                    gt_heat=gt_heat_np,
                    pred_path=pred_path,
                )

        return (
            bce_losses,
            bce_gauss_losses,
            dice_losses,
            dice_gauss_losses,
        )

    def _save_layer_visualizations(
        self,
        out_dir: str,
        sample_id: str,
        layer: int,
        img_bgr: np.ndarray,
        cont_map: np.ndarray,
        gt_heat: np.ndarray,
        pred_path: str,
    ) -> None:
        """Save visualization images for a layer."""
        os.makedirs(out_dir, exist_ok=True)
        # Colorize GT heatmap only
        def colorize01(arr01: np.ndarray) -> np.ndarray:
            x = np.clip(arr01 * 255.0, 0, 255).astype(np.uint8)
            cm = cv2.applyColorMap(x, cv2.COLORMAP_JET)
            return cm

        h, w = gt_heat.shape

        # Predicted visualization
        if pred_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            # Save standalone by writing original pixels as-is (no extra processing)
            pred_img_orig = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
            if pred_img_orig is None:
                # Fallback to rendered from cont_map if read fails
                pred_img_orig = (np.clip(cont_map * 255.0, 0, 255)).astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, f"{sample_id}_layer{layer:02d}_pred_heat.jpg"), pred_img_orig)

            # For side-by-side, ensure size matches heatmap
            pred_side = pred_img_orig
            if pred_side.shape[0] != h or pred_side.shape[1] != w:
                pred_side = cv2.resize(pred_side, (w, h), interpolation=cv2.INTER_LINEAR)
            if pred_side.ndim == 2:
                pred_vis = cv2.cvtColor(pred_side, cv2.COLOR_GRAY2BGR)
            else:
                pred_vis = pred_side
        else:
            # .npy or others: render grayscale from cont_map for visualization
            pred_gray = (np.clip(cont_map * 255.0, 0, 255)).astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, f"{sample_id}_layer{layer:02d}_pred_heat.jpg"), pred_gray)
            pred_vis = cv2.cvtColor(pred_gray, cv2.COLOR_GRAY2BGR)

        # GT heatmap visualization (colorized)
        heat_cm = colorize01(gt_heat)
        cv2.imwrite(os.path.join(out_dir, f"{sample_id}_layer{layer:02d}_gt_heat.jpg"), heat_cm)

        # Side-by-side (pred vs gt heat), no overlay
        side = np.concatenate([pred_vis, heat_cm], axis=1)
        cv2.imwrite(os.path.join(out_dir, f"{sample_id}_layer{layer:02d}_pred-gtheat.jpg"), side)

    def evaluate_dataset(
        self,
        predictions_root: str,
        dataset_root: str,
        sample_ids: Optional[List[str]] = None,
        viz_out_dir: str = "",
        pos_weight_user: float = -1.0,
        pos_weight_gauss_user: float = -1.0,
    ) -> dict:
        """
        Evaluate all samples in the dataset.
        
        Args:
            predictions_root: Root directory containing layer predictions
            dataset_root: Root directory containing ground truth data
            sample_ids: List of sample IDs to evaluate. If None, will auto-discover from layer40
            viz_out_dir: Directory to save visualizations
            pos_weight_user: Positive class weight for BCE
            pos_weight_gauss_user: Positive class weight for BCE_Gauss
            
        Returns:
            Dictionary containing aggregated results
        """
        if sample_ids is None:
            layer40_dir = os.path.join(predictions_root, "layer40")
            sample_ids = self._collect_ids_from_layer_dir(layer40_dir)
            print(f"Found {len(sample_ids)} IDs from {layer40_dir}.")

        all_results = {
            'bce_losses': [],
            'bce_gauss_losses': [],
            'dice_losses': [],
            'dice_gauss_losses': [],
            'sample_ids': []
        }

        for sample_id in sample_ids:
            try:
                (
                    bce_losses,
                    bce_gauss_losses,
                    dice_losses,
                    dice_gauss_losses,
                ) = self.compute_per_layer_losses_for_id(
                    predictions_root=predictions_root,
                    dataset_root=dataset_root,
                    sample_id=sample_id,
                    viz_out_dir=viz_out_dir,
                    pos_weight_user=pos_weight_user,
                    pos_weight_gauss_user=pos_weight_gauss_user,
                )
                
                all_results['bce_losses'].append(bce_losses)
                all_results['bce_gauss_losses'].append(bce_gauss_losses)
                all_results['dice_losses'].append(dice_losses)
                all_results['dice_gauss_losses'].append(dice_gauss_losses)
                all_results['sample_ids'].append(sample_id)
                
            except Exception as e:
                print(f"[Skip] {sample_id}: {e}")
                continue

        return all_results


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--id", type=str, default="", help="Specify a single ID to compute.")
    parser.add_argument("--viz_out", type=str, default="./vis", help="Optional directory to save per-layer visualizations.")
    parser.add_argument("--pos_weight", type=float, default=-1.0, help="Positive class weight for BCE. <=0 to auto-compute (neg/pos).")
    parser.add_argument("--pos_weight_gauss", type=float, default=-1.0, help="Positive class weight for BCE_Gauss. <=0 to auto-compute (sum(1-heat)/sum(heat)).")
    args, _ = parser.parse_known_args()

    predictions_root = "/home/xinyin/qianrui/lmm/UGround_28/uground-13B@reason_seg_val"
    dataset_root = "/home/xinyin/qianrui/lmm/dataset_sesame/reason_seg/ReasonSeg/val"

    # Initialize loss computer
    simi_loss = SimiLoss()

    if args.id:
        ids = [args.id]
    else:
        layer40_dir = os.path.join(predictions_root, "layer40")
        ids = simi_loss._collect_ids_from_layer_dir(layer40_dir)
        print(f"Found {len(ids)} IDs from {layer40_dir}.")

    for sample_id in ids:
        try:
            (
                bce_losses,
                bce_gauss_losses,
                dice_losses,
                dice_gauss_losses,
            ) = simi_loss.compute_per_layer_losses_for_id(
                predictions_root=predictions_root,
                dataset_root=dataset_root,
                sample_id=sample_id,
                viz_out_dir=args.viz_out,
                pos_weight_user=args.pos_weight,
                pos_weight_gauss_user=args.pos_weight_gauss,
            )
        except Exception as e:
            print(f"[Skip] {sample_id}: {e}")
            continue

        bce_str = ", ".join(f"{v:.6f}" for v in bce_losses)
        bce_gauss_str = ", ".join(f"{v:.6f}" for v in bce_gauss_losses)
        dice_str = ", ".join(f"{v:.6f}" for v in dice_losses)
        dice_gauss_str = ", ".join(f"{v:.6f}" for v in dice_gauss_losses)

        print(f"{sample_id} BCE:         [{bce_str}]")
        print(f"{sample_id} BCE_Gauss:   [{bce_gauss_str}]")
        print(f"{sample_id} Dice:        [{dice_str}]")
        print(f"{sample_id} Dice_Gauss:  [{dice_gauss_str}]")


if __name__ == "__main__":
    main() 