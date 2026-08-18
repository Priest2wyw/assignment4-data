import re
from typing import Any

import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.parse.encoding import detect_encoding, bytes_to_str
from resiliparse.extract.html2text import extract_plain_text

from cs336_data.common import get_id_language_model_path


def clear_space(text):
    return re.sub(r"\s+", " ", text).strip()


def extract_text_from_html(html_bytes: bytes) -> str | None:
    encoding_type = detect_encoding(html_bytes)
    html_texts = bytes_to_str(html_bytes, encoding=encoding_type)
    return extract_plain_text(html_texts)


def identify_language(text: str) -> tuple[Any, float]:
    # remove \n
    text = clear_space(text)

    model_path = str(get_id_language_model_path())
    model = fasttext.load_model(model_path)
    lang, score = model.predict(text=text)
    lang = lang[0].split("_")[-1]
    score = score[0]
    return lang, score


def get_html_from_warc(warc_path):
    htmls = []
    urls = []

    # 你当前 fastwarc 版本要求传 file-like object，
    # 不能直接把路径字符串传给 ArchiveIterator
    with open(warc_path, "rb") as f:
        for record in ArchiveIterator(
            f,
            record_types=WarcRecordType.response,
        ):
            # 只保留 HTML response
            content_type = record.http_content_type

            if content_type is None:
                continue

            if "html" not in content_type.lower():
                continue

            url = record.headers.get("WARC-Target-URI")

            # HTTP body，已经不包含 HTTP headers
            html_bytes = record.reader.read()

            if not html_bytes:
                continue

            # 优先使用 HTTP Content-Type 中声明的 charset
            charset = record.http_charset or "utf-8"

            try:
                html = html_bytes.decode(
                    charset,
                    errors="replace",
                )
            except (LookupError, UnicodeDecodeError):
                html = html_bytes.decode(
                    "utf-8",
                    errors="replace",
                )

            htmls.append(html)
            urls.append(url)

    return htmls, urls
