from fastwarc.warc import ArchiveIterator, WarcRecordType
from random import sample
from cs336_data.data_cleaner import get_html_from_warc, identify_language, extract_text_from_html


if __name__ == "__main__":
    warc_path = "/data4/youwei/cs336_data_warc/example.warc.gz"
    htmls, urls = get_html_from_warc(warc_path)
    print(len(htmls), len(urls))

    # -------
    htmls_simple = sample(htmls, 20)
    for html in htmls_simple:
        html_str = extract_text_from_html(bytes(html, encoding="utf-8"))
        lg_cs, score = identify_language(html_str)
        html = html_str[0:200].replace("\n", "").replace(" ", "")
        print(f"lg is {lg_cs},score is {score:.2f} , {html}")
