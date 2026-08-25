import json
from typing import Iterator
from tqdm import tqdm
from pathlib import Path

from cs336_data.data_cleaner import mask_email, mask_ip_address, mask_phone_number, gopher_quality_filter


def get_text(dir_path: Path) -> Iterator[str]:
    for file_path in tqdm(sorted(dir_path.rglob("*wiki_*"))):
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                context = json.loads(line)
                text = context["text"].strip()

                if text:
                    yield text


def process_text(text: str):
    text, _ = mask_email(text)
    text, _ = mask_ip_address(text)
    text, _ = mask_phone_number(text)

    if not gopher_quality_filter(text):
        text = ""
    return text


def main(dir_path: Path, output_file_path: Path, batch_size: int = 2000):
    """
    batch_read_files like "wiki_*" in a dir read line.text
    for text in lines:
        mask_infos
        is_filter by gopher_quality_filter

    write_file:
        __label__{label} {text}\n

    """
    with open(output_file_path, "a") as out_file:
        # get text by generator
        for text in tqdm(get_text(dir_path), desc="开始处理文件"):
            text = process_text(text)
            text = " ".join(text.split("\n"))
            if text.strip():
                out_file.write(f"__label__hq {text}\n")


if __name__ == "__main__":
    enwiki_dir = Path("/data/youwei/files/enwiki/")
    out_file = Path("/data/youwei/files/cs336/cs336_data/CC/enwiki.txt")
    main(enwiki_dir, out_file)
