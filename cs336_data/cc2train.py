import json
import logging
from pathlib import Path

from tqdm import tqdm
from cs336_data.cleaner_enwiki import process_text
from cs336_data.data_cleaner import get_html_from_warc, extract_text_from_html

logger = logging.getLogger(__name__)


def process_one_html(html: str):
    """
    单个html处理
    """
    try:
        html_content = extract_text_from_html(html.encode("utf-8"))
        if not html_content.strip():
            return None
        text = process_text(html_content)

        # 清理换行
        text = " ".join(text.split("\n"))

        return text

    except UnicodeEncodeError:
        logger.warning("html encode failed")

    except Exception:
        logger.exception("process html failed")

    return None


def process_one_warc(file_path: Path):
    """
    单个warc处理
    """
    try:
        htmls, _ = get_html_from_warc(file_path)

    except Exception:
        logger.exception("load warc failed: %s", file_path)
        return []

    texts = []

    for html in tqdm(htmls, total=len(htmls), desc=file_path.name):
        text = process_one_html(html)

        if text:
            texts.append(text)

    return texts


def main(warc_path: Path, out_file_path: Path):

    with out_file_path.open("a", encoding="utf-8") as f:
        for file_path in tqdm(sorted(warc_path.rglob("*warc"))):
            texts = process_one_warc(file_path)

            for text in texts:
                f.write(f"__label__lq {text}\n")


if __name__ == "__main__":
    warc_dir = Path("local-shared-data/CC/cc-main/")
    out_file = Path("/data/youwei/files/cs336/cs336_data/CC/cc_out.txt")
    main(warc_path=warc_dir, out_file_path=out_file)
