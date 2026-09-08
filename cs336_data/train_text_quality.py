import fasttext

from typing import Generator, Optional,Callable, Tuple

from pathlib import Path 
import random
from tqdm import tqdm

def get_lines(file_path:Path):
    print(f"start readlines {file_path}")
    with file_path.open("r", encoding='utf-8') as f:
        lines = f.readlines()
    if len(lines)==0:
        print(f"{file_path} has no lines")
    return lines

def get_train_data(positive_file_path: Path, 
                   negative_file_path: Path, 
                   output_file: str,
                   postive_rate=0.8, 
                   total_size=100_000,
                   seed=42)->list[str]:
    random.seed(seed)
    train_data = []
    negative_lines = get_lines(negative_file_path)
    postive_lines = get_lines(positive_file_path)

    postive_lines_num = min(total_size*(postive_rate), len(postive_lines))
    negative_lines_num = min(total_size*(1- postive_rate), len(negative_lines))

    neg_lines =  random.sample(negative_lines, int(negative_lines_num))
    postive_lines_sample = random.sample(postive_lines, int(postive_lines_num))  
    train_data = neg_lines + postive_lines_sample
    random.shuffle(train_data)
    print(f'postive_lines is {postive_lines_num}, negative_lines is {negative_lines_num}')

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in tqdm(train_data, desc="写入合并文件"):
            f.write(line + '\n')


    return train_data


# ---------- 2. 训练 / 验证划分 ---------- #
def split_train_val(
    merged_file: str,
    train_file: str,
    val_file: str,
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> tuple[int, int]:
    """按 val_ratio 划分训练集与验证集，返回 (训练条数, 验证条数)"""
    random.seed(random_seed)
    lines = [ln.strip() for ln in open(merged_file, 'r', encoding='utf-8') if ln.strip()]
    random.shuffle(lines)
    val_size = int(len(lines) * val_ratio)
    with open(val_file, 'w', encoding='utf-8') as f:
        for ln in tqdm(lines[:val_size], desc="写验证集"):
            f.write(ln + '\n')
    with open(train_file, 'w', encoding='utf-8') as f:
        for ln in tqdm(lines[val_size:], desc="写训练集"):
            f.write(ln + '\n')
    return len(lines) - val_size, val_size

# ---------- 3. 训练 + 评估 ---------- #
def train_quality_classifier(
    train_path: str,
    model_path: str,
    val_path: Optional[str] = None,
    **kwargs
) -> fasttext.FastText:
    """
    训练 fastText 分类器并保存
    默认参数已针对中文/英文维基质量过滤调优，可外部覆盖
    """
    defaults = dict(
        lr=0.1,
        epoch=25,
        wordNgrams=2,
        dim=100,
        loss='softmax',
        minCount=10,
        bucket=2_000_000,
        thread=12,          # 多线程加速
    )
    defaults.update(kwargs)
    print(f"set of train is {defaults}")
    model = fasttext.train_supervised(input=train_path, **defaults)
    model.save_model(model_path)
    if val_path and Path(val_path).exists():
        n, p, r = model.test(val_path)
        print(f"验证集  样本数:{n}  precision:{p:.4f}  recall:{r:.4f}  F1:{2*p*r/(p+r):.4f}")
    return model


def main():
    script_dir = Path("/data/youwei/files/cs336/cs336_data/CC")
    pos_file   = script_dir / "enwiki.txt"
    neg_file   = script_dir / "cc_out.txt"
    merged     = script_dir / "quality_merged.txt"
    train_f    = script_dir / "quality_train.txt"
    val_f      = script_dir / "quality_val.txt"
    model_bin  = script_dir / "quality_classifier.bin"

    if True:
        # 1. 合并 & 打乱
        total = get_train_data(pos_file, neg_file,  str(merged))
        total = len(total)
        if total == 0:
            print("没有有效样本，程序结束")
            return

        # 2. 划分
        train_n, val_n = split_train_val(str(merged), str(train_f), str(val_f))
        print(f"开始训练：训练集 {train_n} 条，验证集 {val_n} 条")

    # 3. 训练
    train_quality_classifier(str(train_f), str(model_bin), str(val_f))

    print(f"模型已保存至 {model_bin}")


if __name__ == "__main__":
    main()



