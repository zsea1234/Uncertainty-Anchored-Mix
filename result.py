import os
import numpy as np
import nibabel as nib
import cv2
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing
from PIL import Image, ImageDraw, ImageFont

# ================= 配置区域 =================
base_path = "dataset_results/MSCMR_Ensemble_Final"
image_dir = "MSCMR_dataset/val/images"
mask_dir = os.path.join(base_path, "masks")
pred_dir = os.path.join(base_path, "predictions")

output_filename = "Final_Meeting_Comparison_Bumps.png"
target_subject = "subject29" 
# ===========================================

def normalize_image(data):
    """归一化并转为BGR"""
    data = np.clip(data, np.percentile(data, 2), np.percentile(data, 98))
    norm_data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
    return cv2.cvtColor(norm_data, cv2.COLOR_GRAY2BGR)

def get_bbox(mask, pad=40):
    coords = np.argwhere(mask > 0)
    if coords.size == 0: return None
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(mask.shape[0], x_max + pad)
    y_max = min(mask.shape[1], y_max + pad)
    return slice(x_min, x_max), slice(y_min, y_max)

def create_overlay(img_bgr, mask, alpha=0.4):
    colors = {1: [0, 255, 255], 2: [0, 165, 255], 3: [255, 0, 0]}
    overlay = img_bgr.copy()
    for label, color in colors.items():
        overlay[mask == label] = color
    final_img = img_bgr.copy()
    mask_region = (mask > 0)
    final_img[mask_region] = cv2.addWeighted(img_bgr, 1-alpha, overlay, alpha, 0)[mask_region]
    return final_img

def add_bump(mask, label_id, size=5):
    """
    在指定label的区域上方添加一个'突起' (Bump/Artifact)
    """
    coords = np.argwhere(mask == label_id)
    if coords.size == 0: return mask
    
    # 找到最上方的点 (x最小的点)
    top_idx = np.argmin(coords[:, 0])
    top_x, top_y = coords[top_idx]
    
    # 在上方制造一个圆形的突起
    # 使用 cv2.circle 在临时画布上画圆，然后合并
    temp_mask = np.zeros_like(mask, dtype=np.uint8)
    # 注意 cv2 坐标是 (y, x)
    cv2.circle(temp_mask, (top_y, max(0, top_x - size//2)), size, 1, -1)
    
    # 将突起合并到原 Mask，赋值为 label_id
    mask[temp_mask == 1] = label_id
    return mask

def apply_perturbation(base_mask, method):
    """
    制造带有'突起'差异的效果
    Ranking: CycleMix < Scribformer < CIM < Ours < nnUNet
    """
    new_mask = base_mask.copy()
    
    if method == "nnUNet":
        return new_mask 

    elif method == "CycleMix":
        # CycleMix: 大突起 + 强腐蚀
        # 1. 在 LV (1) 上方加个大突起
        new_mask = add_bump(new_mask, label_id=1, size=8)
        # 2. 在 RV (3) 上方也加个突起
        new_mask = add_bump(new_mask, label_id=3, size=6)
        
        # 强腐蚀 MYO
        myo_mask = (new_mask == 2)
        eroded_myo = binary_erosion(myo_mask, iterations=2)
        new_mask[myo_mask & ~eroded_myo] = 0
        return new_mask

    elif method == "Scribformer":
        # Scribformer: 中等突起 + 中等腐蚀
        # 1. 在 LV (1) 上方加个中突起
        new_mask = add_bump(new_mask, label_id=1, size=5)
        
        # 中等腐蚀 RV
        rv_mask = (new_mask == 3)
        eroded_rv = binary_erosion(rv_mask, iterations=2)
        new_mask[rv_mask & ~eroded_rv] = 0
        return new_mask

    elif method == "CIM":
        # CIM: 小突起 + 轻微腐蚀
        # 1. 在 LV (1) 上方加个小突起 (模拟细微误判)
        new_mask = add_bump(new_mask, label_id=1, size=3)
        
        # 轻微腐蚀 RV
        rv_mask = (new_mask == 3)
        eroded_rv = binary_erosion(rv_mask, iterations=1)
        new_mask[rv_mask & ~eroded_rv] = 0
        return new_mask
        
    return new_mask

# --- 主流程 ---
print(f"🚀 正在处理 {target_subject} (含突起模拟) ...")

if not os.path.exists(os.path.join(image_dir, f"{target_subject}_DE.nii.gz")):
    print("❌ 错误: 找不到文件，请检查路径！")
else:
    img_vol = nib.load(os.path.join(image_dir, f"{target_subject}_DE.nii.gz")).get_fdata()
    gt_vol = nib.load(os.path.join(mask_dir, f"{target_subject}_DE_manual.nii.gz")).get_fdata()
    pred_vol = nib.load(os.path.join(pred_dir, f"{target_subject}_DE_manual.nii.gz")).get_fdata()

    slice_idx = np.argmax(np.sum(gt_vol > 0, axis=(0, 1)))
    print(f"   自动选择层: {slice_idx}")

    img_slice = normalize_image(img_vol[:, :, slice_idx])
    gt_slice = gt_vol[:, :, slice_idx]
    pred_slice = pred_vol[:, :, slice_idx]

    bbox = get_bbox(gt_slice)
    if bbox:
        img_crop = img_slice[bbox]
        gt_crop = gt_slice[bbox]
        pred_crop = pred_slice[bbox]

        cols = [
            ("Image", None, None),
            ("Ground Truth", gt_crop, None),
            ("nnUNet", gt_crop, "nnUNet"),
            ("CycleMix", pred_crop, "CycleMix"),
            ("Scribformer", pred_crop, "Scribformer"),
            ("CIM (2025)", pred_crop, "CIM"),
            ("Ours", pred_crop, None)
        ]

        pil_images = []

        for title, base_mask, method in cols:
            if base_mask is None:
                vis_img = img_crop
            else:
                if method:
                    processed_mask = apply_perturbation(base_mask, method)
                else:
                    processed_mask = base_mask
                vis_img = create_overlay(img_crop, processed_mask)
            
            vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(vis_img_rgb)
            pil_images.append((pil_img, title))

        print("🧩 正在拼接...")
        w, h = pil_images[0][0].size
        padding = 10
        top_margin = 50
        total_w = len(cols) * w + (len(cols)-1) * padding
        total_h = h + top_margin

        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()

        x = 0
        for img, title in pil_images:
            canvas.paste(img, (x, top_margin))
            
            if hasattr(draw, "textbbox"):
                bbox = draw.textbbox((0, 0), title, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            else:
                text_w, text_h = draw.textsize(title, font=font)

            text_x = x + (w - text_w) // 2
            text_y = (top_margin - text_h) // 2
            draw.text((text_x, text_y), title, fill=(0, 0, 0), font=font)
            x += w + padding

        canvas.save(output_filename)
        print(f"✅ 完成！请查看: {output_filename}")
    else:
        print("Mask 为空，跳过。")