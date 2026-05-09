import os
import argparse
import torch
import numpy as np
import nibabel as nib
from pathlib import Path

# 只导入参数和网络结构
from models import build_model
from main import get_args_parser

def main():
    parser = argparse.ArgumentParser('Pure Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    device = torch.device(args.device)

    # 1. 构建网络结构
    model, _, _, _ = build_model(args)
    model.to(device)
    model.eval()

    # 提取您的 task 名称 (默认是 'MR')
    task_name = list(args.tasks.keys())[0]

    # 2. 我们只盯着 subject29 这一张图
    subject_name = "subject29_DE"
    image_path = f"/home/guest25/zyy/CycleMix/MSCMR_dataset/val/images/{subject_name}.nii.gz"
    
    checkpoints = {
        "BoxInst": {
            "weight": "/data/MSCMR_cycleMix_PU/BoxInst_sim_581.pth",
            "out_dir": "/home/guest25/zyy/results_boxinst/"
        },
        "WeakPolyp": {
            "weight": "/data/MSCMR_cycleMix_PU/WeakPolyp_sim_wide.pth",
            "out_dir": "/home/guest25/zyy/results_weakpolyp/"
        },
        "PA-Seg": {
            "weight": "/data/MSCMR_cycleMix_PU/PA-Seg_sim_638.pth",
            "out_dir": "/home/guest25/zyy/results_paseg/"
        }
    }

    print(f"📖 正在读取图像: {image_path}")
    if not os.path.exists(image_path):
        print(f"❌ 找不到图片，请检查路径: {image_path}")
        return

    # 3. 读取原始数据和空间信息
    img_nii = nib.load(image_path)
    img_data = img_nii.get_fdata()
    affine = img_nii.affine
    header = img_nii.header

    with torch.no_grad():
        for method_name, config in checkpoints.items():
            ckpt_path = config["weight"]
            out_dir = config["out_dir"]

            if not os.path.exists(ckpt_path):
                print(f"⚠️ 跳过 {method_name} (未找到权重: {ckpt_path})")
                continue

            Path(out_dir).mkdir(parents=True, exist_ok=True)
            print(f"🚀 开始纯手动推理: {method_name}")

            # 加载权重
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'])

            # 创建空白的 3D 预测体积容器
            pred_volume = np.zeros_like(img_data, dtype=np.uint8)

            # 4. 逐切片推理
            for z in range(img_data.shape[2]):
                slice_np = img_data[:, :, z]
                
                # 转为 Tensor 并进行标准化
                slice_tensor = torch.from_numpy(slice_np).float()
                
                slice_min, slice_max = slice_tensor.min(), slice_tensor.max()
                if slice_max > slice_min:
                    slice_tensor = (slice_tensor - slice_min) / (slice_max - slice_min)
                
                slice_tensor = (slice_tensor - 0.5) / 0.5 
                
                inputs = slice_tensor.unsqueeze(0).unsqueeze(0).to(device)

                # 🔥 修复：传入 task 名称 🔥
                outputs = model(inputs, task=task_name)

                # 智能剥离模型输出
                if isinstance(outputs, dict):
                    logits = outputs.get('pred_masks', outputs.get('out', list(outputs.values())[0]))
                elif isinstance(outputs, tuple) or isinstance(outputs, list):
                    logits = outputs[0]
                else:
                    logits = outputs

                # 获取分类预测
                pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
                pred_volume[:, :, z] = pred_mask

            # 5. 组合保存
            out_file = os.path.join(out_dir, f"{subject_name}.nii.gz")
            pred_nii = nib.Nifti1Image(pred_volume, affine, header)
            nib.save(pred_nii, out_file)
            print(f"✅ 成功生成并保存: {out_file}\n")

if __name__ == '__main__':
    main()