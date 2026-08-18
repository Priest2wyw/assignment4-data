import re
from typing import Any, Tuple

import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.parse.encoding import detect_encoding, bytes_to_str
from resiliparse.extract.html2text import extract_plain_text

from cs336_data.common import get_id_language_model_path
from cs336_data.common import NSFW_MODEL_PATH, TOXIC_SPEECH_MODEL_PATH


EMAIL_PAT = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PAT = re.compile(
    r"(?<!\d)(?:(?:\+?86[-.\s]?)?1[3-9]\d{9}|(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})(?!\d)"
)
IP_ADDRESS_PAT = r"(\b25[0-5]|\b2[0-4][0-9]|\b[01]?[0-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}"
IP_ADDRESS_PAT = re.compile(IP_ADDRESS_PAT)


EMAIL_MASK = "|||EMAIL_ADDRESS|||"
PHONE_MASK = "|||PHONE_NUMBER|||"
IP_ADDRESS_MASK = "|||IP_ADDRESS|||"


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


def mask_email(context: str) -> Tuple[str, int]:
    replaced_str, number_of_email = None, 0
    replaced_str, number_of_email = EMAIL_PAT.subn(EMAIL_MASK, context)
    return replaced_str, number_of_email


def mask_phone_number(context: str) -> Tuple[str, int]:
    replaced_str, number_of_phone = None, 0
    replaced_str, number_of_phone = PHONE_PAT.subn(PHONE_MASK, context)
    return replaced_str, number_of_phone


def mask_ip_address(context: str) -> Tuple[str, int]:
    replaced_str, number = None, 0
    replaced_str, number = IP_ADDRESS_PAT.subn(IP_ADDRESS_MASK, context)
    return replaced_str, number


def classify_nsfw(text: str) -> Tuple[Any, float]:
    model = fasttext.load_model(str(NSFW_MODEL_PATH))
    label, score = model.predict(text)
    label = label[0].split("_")[-1]
    score = score[0]
    return label, score


def classify_toxic_speech(text: str) -> Tuple[Any, float]:
    model = fasttext.load_model(str(TOXIC_SPEECH_MODEL_PATH))
    label, score = model.predict(text)
    label = label[0].split("_")[-1]
    score = score[0]
    return label, score
