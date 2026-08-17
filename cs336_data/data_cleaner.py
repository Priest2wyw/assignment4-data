import re
from typing import Any

import fasttext
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
