import time
import os
import shutil
import re

# ================= 1. 配置区 =================
# 【请修改为您的真实路径】
LOG_FILE = "train_log.txt"                 # 您跑训练时重定向的日志文件名称
CKPT_DIR = "/data/MSCMR_cycleMix_PU/"         # 必须与您 train.py 的 args.output_dir 保持一致
# ============================================

SOURCE_CKPT = os.path.join(CKPT_DIR, "checkpoint.pth")

# MSCMR 的目标分数及误差范围 (目标分数: (下限, 上限))
TARGETS = {
  "WeakPolyp_sim_wide.pth": (0.580, 0.630)
}

hit_targets = set()

def wait_and_copy(target_name):
    """安全地等待 torch.save 写入完成并进行复制"""
    print("⏳ 检测到目标分数，正在等待模型权重写入磁盘...")
    time.sleep(3) # 给 train.py 里的 torch.save 留出充足的写入时间
    
    # 确保文件大小稳定（没有在被持续写入）
    last_size = -1
    for _ in range(15):
        if os.path.exists(SOURCE_CKPT):
            curr_size = os.path.getsize(SOURCE_CKPT)
            if curr_size == last_size and curr_size > 0:
                break
            last_size = curr_size
        time.sleep(1)
        
    target_path = os.path.join(CKPT_DIR, target_name)
    try:
        shutil.copy2(SOURCE_CKPT, target_path)
        print(f"🎉 成功备份目标权重: {target_path}\n")
    except Exception as e:
        print(f"❌ 备份失败: {e}\n")

def main():
    if not os.path.exists(LOG_FILE):
        print(f"⚠️ 找不到日志文件 {LOG_FILE}，请确认您的训练已经启动并重定向了日志！")
        return

    print(f"🎯 [狙击手已就位] 正在实时监控日志: {LOG_FILE}")
    print(f"🎯 [监控目标] {list(TARGETS.keys())}\n")

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        # 直接跳到文件末尾，只监控启动 sniper.py 之后的新输出
        f.seek(0, 2) 
        
        while len(hit_targets) < len(TARGETS):
            line = f.readline()
            if not line:
                time.sleep(0.5) # 没有新输出时，稍微休息一下
                continue
            
            # 使用正则匹配 "dice score: 0.xxxx"
            match = re.search(r"dice score:\s+(0\.\d+)", line)
            if match:
                current_dice = float(match.group(1))
                
                # 遍历比对目标
                for target_name, (low, high) in TARGETS.items():
                    if target_name in hit_targets:
                        continue
                        
                    if low <= current_dice <= high:
                        print(f"\n🎯 [狙击成功] 命中分数: {current_dice:.4f} (目标: {target_name})")
                        wait_and_copy(target_name)
                        hit_targets.add(target_name)
                        break

    print("✅ 所有目标均已成功捕获！监控程序退出。")

if __name__ == "__main__":
    main()