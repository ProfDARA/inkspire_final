# file: versi3/lexicon/score_text_metrics.py
import argparse, re, os
from pathlib import Path
import numpy as np, pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# ======= (opsional) Sastrawi untuk stemming =======
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    _SASTRAWI_OK = True
except Exception:
    _SASTRAWI_OK = False

# ===================== Kamus =====================
STOPWORDS = set(
    "yang di ke dari kepada untuk dengan pada oleh demi agar supaya sehingga karena namun tetapi sedangkan atau dan serta "
    "pun bila jika ketika maka lalu kemudian yaitu yakni adalah ialah seperti sebagai dalam terhadap antara tanpa bukan "
    "ini itu tersebut para nya saya aku kami kita kamu anda beliau dia mereka sini sana situ hal bahwa hingga setiap seluruh "
    "beberapa berbagai sudah belum masih jangan walau walaupun meski meskipun sebaliknya selainnya sementara selanjutnya berikutnya "
    "sehabis seusai setelah sebelum sesudah karenanya akibatnya oleh karena itu contohnya misalnya misal".split()
)

CONNECTIVES = {
    "karena","sebab","akibatnya","sehingga","maka","oleh","oleh karena itu","oleh sebab itu","disebabkan oleh",
    "yang menyebabkan","yang mengakibatkan","dengan demikian","sebagai akibat","konsekuensinya","sebagai konsekuensi",
    "dampaknya","implikasinya","berkat","namun","tetapi","meski","meskipun","walau","sebaliknya","padahal","berbeda dengan",
    "di sisi lain","kendati demikian","walaupun demikian","akan tetapi","namun demikian","meskipun demikian","sekalipun demikian",
    "bertolak belakang dengan","kemudian","lalu","selanjutnya","berikutnya","sebelumnya","setelah","sebelum","setelah itu",
    "pada tahap berikutnya","pada akhirnya","selama","ketika","saat","seiring dengan","hingga","sementara itu","pada saat yang sama",
    "di kemudian hari","sepanjang","sejak itu","di awal","di akhir","misal","misalnya","contohnya","sebagai contoh","yakni","yaitu",
    "dengan kata lain","antara lain","seperti","termasuk","sebagai ilustrasi","contoh kasus","khususnya","selain itu","di samping itu",
    "lebih lanjut","lebih jauh","lagipula","sebagai kesimpulan","kesimpulannya","dapat disimpulkan bahwa","ringkasnya","singkatnya",
    "dari paparan tersebut","jika","apabila","bila","seandainya","asalkan","jikalau","andaikan","kecuali jika","agar","supaya","guna",
    "dalam rangka","dibandingkan dengan","dibanding dengan","seperti halnya","serupa dengan","berkebalikan dengan","didefinisikan sebagai",
    "yang dimaksud dengan","dalam konteks ini","dalam hal ini","berdasarkan","menurut","mengacu pada","merujuk pada","terbatas pada",
    "hanya saja","sebatas"
}

ACTION_VERBS = {
    "pergi","datang","pulang","berjalan","berlari","melompat","masuk","keluar","naik","turun",
    "menjelaskan","meneliti","mengumpulkan","menyimpulkan","membuat","menulis","makan","minum","tidur",
    "mencapai","meningkatkan","menurunkan","menyaring","mengurutkan","menyalurkan"
}
TIME_WORDS = {"kemarin","tadi","besok","nanti","sekarang","pada","pukul","jam","minggu","bulan","tahun",
              "januari","februari","maret","april","mei","juni","juli","agustus","september","oktober","november","desember",
              "hari","senin","selasa","rabu","kamis","jumat","sabtu","minggu"}
PLACE_PREPS = {"di","ke","dari"}
PASSIVE_HINTS = {"oleh"}

# ===================== Util & Preprocess =====================
SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-žΑ-ωА-яİıĞğŞşÇçŊŋ']+")
VOWELS = set("aiueoAIUEO")

META_PAT = re.compile(
    r"(?:^\s*ABSTRAK\b|SKRIPSI|PROGRAM STUDI|FAKULTAS|UNIVERSITAS|Pembimbing|^\d{4}\.?$)",
    re.IGNORECASE
)

NUM_RE = re.compile(r"""(?ix)
(?:\d{1,3}([.,\s]\d{3})+[.,]\d+|\d+[.,]\d+|\d+)
""")

def clean_text(s: str) -> str:
    s = str(s)
    s = re.sub(r'""', '"', s)                 # rapikan kutip ganda
    s = re.sub(r"(\d)\s+(\d)", r"\1\2", s)    # "0 58" -> "058"
    s = NUM_RE.sub(" NUM ", s)                # angka -> NUM
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_metadata(line: str) -> bool:
    if META_PAT.search(line): return True
    toks = WORD_RE.findall(line)
    return len(toks) < 3

# ---- Tokenization (plain & stemmed) ----
def tokens_plain(text): 
    return [w.lower() for w in WORD_RE.findall(str(text))]

def build_stemmer():
    if not _SASTRAWI_OK:
        return None
    try:
        factory = StemmerFactory()
        return factory.create_stemmer()
    except Exception:
        return None

def tokens_stem(text, stemmer=None):
    toks = [w.lower() for w in WORD_RE.findall(str(text))]
    if stemmer is None:  # fallback aman
        return [t for t in toks if t not in STOPWORDS]
    out = []
    for t in toks:
        if t in STOPWORDS or t == "num":
            out.append(t)  # biarkan stopword/NUM apa adanya (NUM tetap num)
        else:
            out.append(stemmer.stem(t))
    # buang stopword setelah stemming juga
    return [t for t in out if t not in STOPWORDS]

# fungsi pointer yang akan di-set di main()
def TOKENS(text): 
    return tokens_plain(text)

def ratio_in_vocab(tok, vocab):
    n = len(tok) or 1
    return sum(t in vocab for t in tok)/n

def syllable_guess(word):
    c, pv = 0, False
    for ch in str(word):
        v = ch in VOWELS
        if v and not pv: c += 1
        pv = v
    return max(1, c)

def split_sentences(text):
    s = str(text).strip()
    if not s: return []
    sents = [t.strip() for t in SENT_SPLIT_RE.split(s) if t.strip()]
    return sents if sents else [s]

def clamp01(x): return float(np.clip(x, 0.0, 1.0))
def clamp100(x): return float(np.clip(x, 0.0, 100.0))

# ---- Flesch: RAW (bisa <0 / >100), normalisasi dilakukan setelah seluruh DF terkumpul
def flesch_raw(text: str) -> float:
    sents = split_sentences(text)
    n_sent = max(1, len(sents))
    tok = TOKENS(text)                 # <- gunakan token fungsi aktif (plain/stem)
    n_words = max(1, len(tok))
    # NOTE: suku kata dihitung dari string asli (tanpa stem) supaya lebih natural
    #       tapi bisa juga dari token; pakai string asli agar stabil.
    words_for_syll = tokens_plain(text)
    n_syll = sum(syllable_guess(w) for w in words_for_syll)
    return 206.835 - 1.015*(n_words/n_sent) - 84.6*(n_syll/n_words)

def concretize_tokens_0_100(tok, conc_map):
    vals = [float(conc_map[t]) for t in tok if t in conc_map]
    if not vals: return (np.nan, np.nan, np.nan)
    mn, mx = min(vals), max(vals)
    if 0.0 <= mn <= 5.0 and mx <= 5.0 and mn == 0.0:
        scaled = [v/5.0*100.0 for v in vals]             # 0..5
    elif 1.0 <= mn <= 5.0 and mx <= 5.0:
        scaled = [((v-1.0)/4.0)*100.0 for v in vals]     # 1..5
    else:
        a,b = (mn, mx) if mx>mn else (0.0,1.0)
        scaled = [((v-a)/(b-a))*100.0 for v in vals]
    mean100 = float(np.mean(scaled))
    conc_ratio = mean100/100.0
    return (mean100, conc_ratio, 1.0 - conc_ratio)

def syntactic_simplicity(text):
    tok = TOKENS(text)
    n_tok = len(tok)
    if n_tok == 0: return 100.0
    avg_wlen = np.mean([len(t) for t in tok])
    punct_density = sum(ch in ",;:" for ch in text)/max(1, len(text))
    clause_ratio = ratio_in_vocab(tok, {"yang","bahwa","ketika","jika","apabila","sehingga","karena","walaupun","meskipun"})
    len_term = 1.0 - np.tanh((n_tok-12)/30)
    wlen_term = 1.0 - np.tanh((avg_wlen-5.0)/4.0)
    punct_term = 1.0 - 4.0*punct_density
    clause_term = 1.0 - 2.0*clause_ratio
    score = 100*np.clip(0.35*len_term + 0.25*wlen_term + 0.20*punct_term + 0.20*clause_term, 0, 1)
    return clamp100(score)

def narrativity(text, conc_ratio):
    tok = TOKENS(text)
    pron = ratio_in_vocab(tok, {"aku","saya","kami","kita","kamu","dia","mereka","dirinya"})
    act  = ratio_in_vocab(tok, ACTION_VERBS)
    time = ratio_in_vocab(tok, TIME_WORDS)
    place= ratio_in_vocab(tok, PLACE_PREPS)
    passive = ratio_in_vocab(tok, PASSIVE_HINTS)
    conc = conc_ratio if not np.isnan(conc_ratio) else 0.5
    score = 100*np.clip(0.25*conc + 0.20*pron + 0.20*act + 0.10*time + 0.10*place - 0.05*passive, 0, 1)
    return clamp100(score)

# ===================== Main =====================
def main():
    ap = argparse.ArgumentParser(description="Skoring per kalimat: narativitas, cohesion, syntactic, Flesch; dengan pre-clean, stemming opsional & RTX.")
    ap.add_argument("--kata-csv", required=True, help="CSV peta kata → concreteness (Kata,concreteness_pred,...)")
    ap.add_argument("--raw-csv", required=True, help="CSV kalimat (Kalimat,Judul,Penulis,Jurusan)")
    ap.add_argument("--out", default="output_naratif.csv", help="Output CSV")
    ap.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2", help="SentenceTransformer")
    ap.add_argument("--batch-size", type=int, default=512, help="Batch size encoding")
    ap.add_argument("--min-tokens", type=int, default=4, help="Hapus baris < min token")
    ap.add_argument("--max-tokens", type=int, default=60, help="Hapus baris > max token")
    ap.add_argument("--drop-conc-cols", action="store_true", help="Hilangkan kolom concreteness di output")
    # stemming ON by default; pakai --no-stem untuk mematikan
    ap.add_argument("--no-stem", dest="use_stem", action="store_false", help="Matikan stemming Sastrawi")
    ap.set_defaults(use_stem=True)
    args = ap.parse_args()

    # set token function (plain / stem)
    global TOKENS
    stemmer = None
    if args.use_stem:
        if not _SASTRAWI_OK:
            print("[WARN] Sastrawi tidak terpasang. Jalankan: pip install Sastrawi  — fallback ke tokens plain.")
            TOKENS = tokens_plain
        else:
            stemmer = build_stemmer()
            if stemmer is None:
                print("[WARN] Gagal inisialisasi Sastrawi. Fallback ke tokens plain.")
                TOKENS = tokens_plain
            else:
                TOKENS = lambda text: tokens_stem(text, stemmer=stemmer)
    else:
        TOKENS = tokens_plain

    # 1) peta concreteness
    df_k = pd.read_csv(args.kata_cvv if hasattr(args, "kata_cvv") else args.kata_csv, encoding="utf-8-sig", skipinitialspace=True)
    df_k.columns = [c.strip() for c in df_k.columns]
    if "Kata" not in df_k.columns or "concreteness_pred" not in df_k.columns:
        raise ValueError("File kata harus punya kolom 'Kata' dan 'concreteness_pred'.")
    conc_map = {str(k).strip().lower(): float(v) for k, v in zip(df_k["Kata"], df_k["concreteness_pred"]) if pd.notna(k) and pd.notna(v)}

    # 2) data kalimat
    df = pd.read_csv(args.raw_csv, encoding="utf-8-sig", skipinitialspace=True, engine="python", on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    for col in ["Kalimat","Judul","Penulis","Jurusan"]:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di raw-csv.")
    df["_row_idx"] = np.arange(len(df))

    # bersihkan + filter metadata/aneh
    df["Kalimat_clean"] = df["Kalimat"].astype(str).map(clean_text)
    word_counts = df["Kalimat_clean"].map(lambda s: len(WORD_RE.findall(s)))
    keep_mask = (~df["Kalimat_clean"].map(is_metadata)) & (word_counts >= args.min_tokens) & (word_counts <= args.max_tokens)
    df = df[keep_mask].copy()

    # 3) embedder SBERT (pakai RTX kalau ada)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(args.model, device=device)

    out_rows = []
    df_sorted = df.sort_values(["Judul","_row_idx"], kind="stable")
    for judul, g in df_sorted.groupby("Judul", sort=False):
        idxs = list(g.index)
        sents_raw = [str(df_sorted.loc[i, "Kalimat"]) for i in idxs]
        sents = [str(df_sorted.loc[i, "Kalimat_clean"]) for i in idxs]

        # ===== precompute per judul =====
        E = embedder.encode(sents, batch_size=args.batch_size, convert_to_numpy=True, normalize_embeddings=True)
        sims_adj = (E[:-1] * E[1:]).sum(axis=1) if len(E) > 1 else np.array([])

        toks_list = [TOKENS(t) for t in sents]
        sets_nsw = [set(w for w in tt if w not in STOPWORDS) for tt in toks_list]
        overlaps = []
        for k in range(len(sets_nsw)-1):
            a, b = sets_nsw[k], sets_nsw[k+1]
            overlaps.append((len(a & b)/len(a | b)) if a and b else 0.0)

        texts_lower = [t.lower() for t in sents]
        conn_hit = [any(c in tl for c in CONNECTIVES) for tl in texts_lower]
        conn_ratio = [max(ratio_in_vocab(tt, CONNECTIVES), 0.05 if conn_hit[i] else 0.0) for i, tt in enumerate(toks_list)]

        for pos, i in enumerate(idxs):
            kal_raw = sents_raw[pos]          # simpan apa adanya ke output
            kal = sents[pos]                  # versi bersih untuk hitung
            tt  = toks_list[pos]
            mean100, conc_ratio, abs_ratio = concretize_tokens_0_100(tt, conc_map)

            prev_sim = sims_adj[pos-1] if pos-1 >= 0 and len(sims_adj)>0 else None
            next_sim = sims_adj[pos]   if pos < len(sims_adj) else None
            prev_ov  = overlaps[pos-1] if pos-1 >= 0 and len(overlaps)>0 else None
            next_ov  = overlaps[pos]   if pos < len(overlaps) else None

            sims = [x for x in (prev_sim, next_sim) if x is not None]
            ovs  = [x for x in (prev_ov,  next_ov)  if x is not None]

            refc = 100*clamp01(0.55*(np.mean(ovs) if ovs else 0.0) + 0.45*(np.mean(sims) if sims else 0.0))
            deep = 100*clamp01(0.6*conn_ratio[pos] + 0.4*(np.mean(sims) if sims else 0.0))

            syns = syntactic_simplicity(kal)
            narr = narrativity(kal, conc_ratio)
            fre_raw = flesch_raw(kal)   # <-- RAW

            out_rows.append({
                "_row_idx": df_sorted.loc[i, "_row_idx"],
                "Kalimat": kal_raw,  # tampilkan teks asli
                "Judul": df_sorted.loc[i, "Judul"],
                "Penulis": df_sorted.loc[i, "Penulis"],
                "Jurusan": df_sorted.loc[i, "Jurusan"],
                "Word_Concreteness_Mean_100": None if np.isnan(mean100) else round(mean100, 2),
                "Concrete_Ratio": None if np.isnan(conc_ratio) else round(conc_ratio, 4),
                "Abstract_Ratio": None if np.isnan(conc_ratio) else round(1.0 - conc_ratio, 4),
                "Narrativity": round(narr, 2),
                "Referential_Cohesion": round(refc, 2),
                "Deep_Cohesion": round(deep, 2),
                "Syntactic_Simplicity": round(syns, 2),
                "FRE_raw": round(float(fre_raw), 2),   # simpan Flesch asli
            })

    out_df = pd.DataFrame(out_rows).sort_values("_row_idx", kind="stable").drop(columns=["_row_idx"])

    # === Rescale Flesch ke 0–100 berdasarkan distribusi korpus (robust p5–p95) ===
    fre = out_df["FRE_raw"].to_numpy()
    if len(fre) >= 10:  # cukup data untuk percentile
        p5, p95 = np.nanpercentile(fre, [5, 95])
        den = max(1e-6, p95 - p5)
        out_df["Flesch_Normalized"] = np.clip(100*(fre - p5)/den, 0, 100).round(2)
    else:
        out_df["Flesch_Normalized"] = np.clip(fre, 0, 100).round(2)

    # opsional: drop kolom concreteness di output (sisakan mean biar informatif)
    if args.drop_conc_cols:
        out_df = out_df.drop(columns=["Concrete_Ratio","Abstract_Ratio"], errors="ignore")

    # simpan
    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[OK] Saved → {out_path} (rows={len(out_df)}) | device={device} | stem={'ON' if args.use_stem and _SASTRAWI_OK else 'OFF'}")

if __name__ == "__main__":
    main()
