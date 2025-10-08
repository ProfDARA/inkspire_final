# path: scripts/words_by_jurusan.py
import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

DEFAULT_INPUT = "versi3/db/rawdata_clean.csv"
TEXT_COL = "Kalimat"
LABEL_COL = "Jurusan"
DEFAULT_MIN_LEN = 2
DEFAULT_TOP_K = 0  # 0=semua

STOPWORDS_ID = {
    "yang","dan","di","ke","dari","pada","untuk","dengan","atau","itu","ini","ada",
    "bahwa","sebagai","dalam","terhadap","antara","para","oleh","akan","bagi","juga",
    "tidak","tersebut","sebuah","sudah","lebih","agar","dapat","adalah","kita","kami",
    "saya","ia","dia","mereka","kamu","serta","pun","sehingga","setelah","sebelum",
    "saat","ketika","tahun","bulan","hari","guna","universitas","skripsi","prodi","fakultas"
}

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", re.UNICODE)
_NUMERIC_RE = re.compile(r"^\d+$")

def fix_mojibake(s: Optional[str]) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    t = str(s)
    repl = {"â€œ":'"',"â€":'"',"â€˜":"'", "â€™":"'", "â€“":"-","â€”":"-","â€¢":"*","â€¦":"...","Â ":" ","Â":"","Ã—":"×"}
    for a,b in repl.items(): t = t.replace(a,b)
    return re.sub(r"\s+"," ",t).strip()

def tokenize(text: str, stopwords: set, min_len: int) -> List[str]:
    toks = [t.lower() for t in _TOKEN_RE.findall(fix_mojibake(text))]
    out = []
    for t in toks:
        if len(t) < min_len: continue
        if t in stopwords: continue
        if _NUMERIC_RE.fullmatch(t): continue
        out.append(t)
    return out

def count_over(texts: Iterable[str], stopwords: set, min_len: int) -> Counter:
    c = Counter()
    for txt in texts:
        c.update(tokenize(txt, stopwords, min_len))
    return c

def build_words_report(df: pd.DataFrame, min_len: int, top_k: int, extra_stop: Optional[Sequence[str]]) -> pd.DataFrame:
    stop = set(STOPWORDS_ID)
    if extra_stop:
        stop |= {w.strip().lower() for w in extra_stop if w.strip()}

    # per Jurusan
    rows = []
    for jur, g in df.groupby(LABEL_COL, dropna=False):
        jurusan = "" if pd.isna(jur) else str(jur)
        cnt = count_over(g[TEXT_COL].astype(str).tolist(), stop, min_len)
        items = cnt.most_common(top_k or None)
        for w, c in items:
            rows.append((w, c, jurusan))

    # agregat ALL
    all_cnt = count_over(df[TEXT_COL].astype(str).tolist(), stop, min_len)
    all_items = all_cnt.most_common(top_k or None)
    for w, c in all_items:
        rows.append((w, c, "ALL"))

    out = pd.DataFrame(rows, columns=["Kata", "Frekuensi", "Jurusan"])
    out = out.sort_values(["Jurusan", "Frekuensi"], ascending=[True, False]).reset_index(drop=True)
    return out

def main():
    ap = argparse.ArgumentParser(description="Hitung frekuensi kata berurutan: Kata, Frekuensi, Jurusan")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="Path rawdata_clean.csv")
    ap.add_argument("--min-len", type=int, default=DEFAULT_MIN_LEN, help="Minimal panjang kata")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-K per jurusan (0=semua)")
    ap.add_argument("--stopwords", default="", help="Tambahan stopword, pisah koma")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(f"Tidak menemukan file: {src.resolve()}")

    df = pd.read_csv(src, encoding="utf-8-sig")
    if TEXT_COL not in df.columns:
        raise KeyError(f"Kolom '{TEXT_COL}' tidak ada. Kolom: {list(df.columns)}")
    if LABEL_COL not in df.columns:
        df[LABEL_COL] = ""

    extra = [w for w in args.stopwords.split(",")] if args.stopwords else None
    report = build_words_report(df, args.min_len, args.top_k, extra)

    out_path = src.with_name(src.stem + "_kata_jurusan.csv")
    report.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Sukses -> {out_path}")
    print(report.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
