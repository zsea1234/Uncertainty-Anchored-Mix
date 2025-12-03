#!/bin/bash

# 设置你的数据目录路径 (根据你的实际情况修改)
TARGET_DIR="/data/MSCMR_cycleMix_PU"

echo "正在清理目录: $TARGET_DIR"

# 1. 删除所有以 'checkpoint' 开头且以数字结尾的 .pth 文件 (如 checkpoint0099.pth)
# 这些通常是定期的中间保存点
find "$TARGET_DIR" -name "checkpoint[0-9]*.pth" -type f -print -delete

# 2. 删除所有以 'new_checkpoint.pth' 结尾的文件 (如 0.8020new_checkpoint.pth)
# 如果你只想保留分数最高的那个 (假设你已经手动备份了最好的)，可以取消注释下面这行
# 注意：这会删除所有形如 0.xxxxnew_checkpoint.pth 的文件，请确保你已经记住了最好的那个分数或文件
# find "$TARGET_DIR" -name "*new_checkpoint.pth" -type f ! -name "0.8047new_checkpoint.pth" -print -delete
find "$TARGET_DIR" -name "*.pth" -type f ! -name "best_checkpoint.pth" ! -name "checkpoint.pth" -print -delete

echo "清理完成！"
echo "当前剩余文件："
ls -lh "$TARGET_DIR"/*.pth