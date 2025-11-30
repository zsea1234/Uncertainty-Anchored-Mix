import SimpleITK as sitk
import numpy as np
from pathlib import Path
import torch
from torch.utils import data
import nibabel as nib
from . import transforms as T

def load_nii(img_path):
    nimg = nib.load(img_path)
    # get_data 已弃用; 使用 dataobj 以保留原始数据类型语义
    return np.asanyarray(nimg.dataobj), nimg.affine, nimg.header

class mscmrSeg(data.Dataset):
    def __init__(self, img_folder, lab_folder, lab_values, transforms):
        self._transforms = transforms
        # 获取所有图片和标签路径
        img_paths = sorted(list(img_folder.iterdir()))
        lab_paths = sorted(list(lab_folder.iterdir()))
        self.lab_values = lab_values
        self.examples = []
        self.img_dict = {}
        self.lab_dict = {}

        # 修改逻辑：不再使用 zip 强行配对，而是基于文件名进行匹配
        for img_path in img_paths:
            # 修复：处理 .nii.gz 文件的 stem 问题
            # 如果文件是 subject13_DE.nii.gz, stem 是 subject13_DE.nii，这会导致匹配失败
            img_name = img_path.stem
            if img_name.endswith('.nii'):
                img_name = img_name[:-4]
            
            # 在标签列表中寻找所有以该图片名开头的文件，但排除包含 "dense" 的文件
            # 例如: subject13_DE 只匹配 subject13_DE_scribble，忽略 subject13_DE_dense
            matching_labels = [p for p in lab_paths if p.name.startswith(img_name) and "dense" not in p.stem]
            
            if not matching_labels:
                print(f"警告: 未找到图片 {img_name} 的对应标签 (已排除 dense)")
                continue

            # 读取图片 (只读一次)
            img = self.read_image(str(img_path))
            self.img_dict.update({img_name: img})

            # 遍历找到的所有匹配标签并分别创建配对
            for lab_path in matching_labels:
                lab_name = lab_path.stem
                if lab_name.endswith('.nii'):
                    lab_name = lab_name[:-4]
                    
                print(f"配对成功: {img_name} <-> {lab_name}")
                
                lab = self.read_label(str(lab_path))
                self.lab_dict.update({lab_name: lab})

                # 验证形状 (使用 img[0] 因为 read_image 返回的是 [data, scale])
                assert img[0].shape[2] == lab[0].shape[2], \
                    f"形状不匹配: {img_name} ({img[0].shape}) vs {lab_name} ({lab[0].shape})"

                # 生成切片索引 (此处保持原逻辑，只沿 Z 轴切片)
                self.examples += [(img_name, lab_name, -1, -1, slice) for slice in range(img[0].shape[2])]

    def __getitem__(self, idx):
        img_name, lab_name, Z, X, Y = self.examples[idx]

        if Z != -1:
            img = self.img_dict[img_name][0][Z, :, :]
            lab = self.lab_dict[lab_name][0][Z, :, :]
        elif X != -1:
            img = self.img_dict[img_name][0][:, X, :]
            lab = self.lab_dict[lab_name][0][:, X, :]
        elif Y != -1:
            img = self.img_dict[img_name][0][:, :, Y]
            scale_vector_img = self.img_dict[img_name][1]
            
            lab = self.lab_dict[lab_name][0][:, :, Y]
            scale_vector_lab = self.lab_dict[lab_name][1]
        else:
            raise ValueError(f'无效索引: ({Z}, {X}, {Y})')

        img = np.expand_dims(img, 0)
        lab = np.expand_dims(lab, 0)
        
        target = {'name': lab_name, 'slice': (Z, X, Y), 'masks': lab, 'orig_size': lab.shape}

        if self._transforms is not None:
            # 注意: scale_vector_img 仅在 Y 切片逻辑块中定义。
            # 目前看起来只用了 Y 切片，所以是安全的。
            img, target = self._transforms([img, scale_vector_img], [target, scale_vector_lab])

        return img, target

    def read_image(self, img_path):
        img_dat = load_nii(img_path)
        img = img_dat[0]
        pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
        target_resolution = (1.36719, 1.36719)
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])
        img = img.astype(np.float32)
        return [(img - img.mean()) / img.std(), scale_vector]

    def read_label(self, lab_path):
        lab_dat = load_nii(lab_path)
        lab = lab_dat[0]
        pixel_size = (lab_dat[2].structarr['pixdim'][1], lab_dat[2].structarr['pixdim'][2])
        target_resolution = (1.36719, 1.36719)
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])
        return [lab, scale_vector]

    def __len__(self):
        return len(self.examples)


def make_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize()
    ])

    if image_set == 'train':
        return T.Compose([
            T.Rescale(),
            T.RandomHorizontalFlip(),
            T.RandomRotate((0, 360)),
            T.PadOrCropToSize([212, 212]),
            normalize,
        ])
    
    if image_set == 'val':
        return T.Compose([
            T.Rescale(),
            T.PadOrCropToSize([212, 212]),
            normalize
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    repo_root = Path(__file__).resolve().parents[1]
    root = repo_root / args.dataset
    assert root.exists(), f'提供的 MSCMR 路径 {root} 不存在'
    
    PATHS = {
        "train": (root / "train" / "images", root / "train" / "labels"),
        "val": (root / "val" / "images", root / "val" / "labels"),
    }

    img_folder, lab_folder = PATHS[image_set]
    dataset_dict = {}
    
    for task, value in args.tasks.items():
        img_task, lab_task = img_folder, lab_folder
        lab_values = value['lab_values']
        dataset = mscmrSeg(img_task, lab_task, lab_values, transforms=make_transforms(image_set))
        dataset_dict.update({task : dataset})
        
    return dataset_dict