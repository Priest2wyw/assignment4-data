import hashlib
from collections import defaultdict


def _line_hash(line: str) -> str:
    """
    对一行文本计算稳定哈希值
    """
    return hashlib.md5(line.encode("utf-8")).hexdigest()


def exact_deduplication(file_paths: list[str]) -> None:
    """
    对给定文件列表执行精确行去重：
    - 第一次遍历：统计每一行（基于哈希）的全局出现次数
    - 第二次遍历：仅保留全语料中唯一出现的行，重写文件

    参数：
        file_paths: 输入文件路径列表
    """

    # ---------- 第一次遍历：统计 ----------
    hash_counter = defaultdict(int)

    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                h = _line_hash(line)
                hash_counter[h] += 1

    # ---------- 第二次遍历：重写 ----------
    for path in file_paths:
        unique_lines = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                h = _line_hash(line)
                if hash_counter[h] == 1:
                    unique_lines.append(line)

        # 覆盖写回文件
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(unique_lines)
