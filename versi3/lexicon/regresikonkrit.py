# file: predict_concreteness_id.py
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# ---------- Konstanta model ----------
MODEL_FILE = "concreteness_regressor.joblib"
META_FILE = "concreteness_meta.json"

# ---------- Utils umum ----------
def _norm(s: str) -> str:
    return str(s).strip().casefold()

def find_col(cols, candidates):
    """
    Temukan kolom dengan toleransi huruf besar/kecil & spasi.
    candidates: iterable nama kandidat (case-insensitive).
    """
    cand = {_norm(c) for c in candidates}
    for c in cols:
        if _norm(c) in cand:
            return c
    raise KeyError(f"Tidak menemukan kolom dari kandidat: {candidates}. Kolom tersedia: {list(cols)}")

def clean_id_word(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

# ---------- IO ----------
def load_english_lexicon(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    col_word = find_col(df.columns, {"word"})
    col_conc = find_col(df.columns, {"conc.m", "conc_mean", "concreteness", "concreteness_mean"})

    df = df[[col_word, col_conc]].rename(columns={col_word: "word_en", col_conc: "conc"})
    df["word_en"] = df["word_en"].astype(str).str.lower().str.strip()
    df["conc"] = pd.to_numeric(df["conc"], errors="coerce")
    df = df.dropna(subset=["word_en", "conc"]).drop_duplicates(subset=["word_en"])
    if len(df) < 1000:
        raise ValueError("Lexicon Inggris terlalu kecil/kolom salah. Pastikan file benar.")
    return df

def load_id_words_csv(path, input_col=None):
    # skipinitialspace mengabaikan spasi setelah koma → header "Kata ,Frekuensi" tetap aman
    df = pd.read_csv(path, encoding="utf-8-sig", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    # kandidat umum + “Kata”
    default_candidates = {"Kata", "kata", "word_id", "word", "term", "token", "teks"}
    if input_col is None:
        try:
            input_col = find_col(df.columns, default_candidates)
        except Exception:
            raise KeyError("Tidak menemukan kolom kata di CSV input. Gunakan --input-col.")
    else:
        match = [c for c in df.columns if _norm(c) == _norm(input_col)]
        if not match:
            raise KeyError(f"Kolom '{input_col}' tidak ada. Kolom tersedia: {list(df.columns)}")
        input_col = match[0]

    df[input_col] = df[input_col].map(clean_id_word)
    df = df.dropna(subset=[input_col])
    df = df[df[input_col].str.len() > 0]
    return df, input_col

# ---------- Model concreteness ----------
def train_and_save_regressor(lex_df, model_name, batch_size=256):
    embedder = SentenceTransformer(model_name)
    X = embedder.encode(
        lex_df["word_en"].tolist(),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    y = lex_df["conc"].astype(float).values
    reg = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("ridge", Ridge(alpha=1.0, random_state=42)),
    ])
    reg.fit(X, y)
    joblib.dump(reg, MODEL_FILE)
    Path(META_FILE).write_text(json.dumps({"embedder": model_name}), encoding="utf-8")
    print(f"[OK] Model saved → {MODEL_FILE}, meta → {META_FILE}")
    return embedder, reg

def load_regressor(model_file=MODEL_FILE, meta_file=META_FILE):
    reg = joblib.load(model_file)
    meta = json.loads(Path(meta_file).read_text(encoding="utf-8"))
    embedder = SentenceTransformer(meta["embedder"])
    return embedder, reg

def ensure_model(lex_en_path, model_name, batch_size):
    """
    Pakai model cache jika ada; jika belum ada → train sekali lalu simpan.
    """
    if Path(MODEL_FILE).exists() and Path(META_FILE).exists():
        try:
            embedder, reg = load_regressor()
            print("[INFO] Cached regressor loaded.")
            return embedder, reg
        except Exception as e:
            print(f"[WARN] Gagal load model cache: {e}. Retrain...")

    # Train jika cache tidak tersedia
    lex_en = load_english_lexicon(lex_en_path)
    print(f"[INFO] English lexicon loaded: {len(lex_en)} entries.")
    embedder, reg = train_and_save_regressor(lex_en, model_name=model_name, batch_size=batch_size)
    return embedder, reg

def predict_concreteness(embedder, reg, words, batch_size=256):
    X = embedder.encode(
        words,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    yhat = reg.predict(X)
    return np.clip(yhat, 1.0, 5.0)  # skala 1–5 sesuai Brysbaert

# ---------- Heuristik & fitur teks untuk 3 metrik ----------
# Daftar minimalis tapi efektif (boleh kamu perluas nanti)
PRONOUNS = {"aku","saya","kami","kita","kamu","engkau","dia","ia","mereka","dirinya"}
ACTION_VERBS = {
    "pergi","datang","melihat","berkata","berlari","mencari","membawa","mendengar","menjawab",
    "bertemu","menyelamatkan","membuka","menutup","mendorong","menarik","menulis","membaca",
    "membeli","menjual","membuat","mengambil","memberi","berjalan","berbicara","bercerita"
}
TEMP_CONN = {"kemudian","lalu","selanjutnya","akhirnya","sebelumnya","pada","pada akhirnya","sesudahnya"}
CAUSAL_CONN = {
    "karena","sebab","sehingga","maka","oleh karena itu","akibatnya","dengan demikian","konsekuensinya",
    "namun","meskipun","walaupun","sebaliknya"
}
# stopword ringkas (bisa ganti ke daftar lengkap jika perlu)
ID_STOP = {
    "yang","dan","di","ke","dari","ini","itu","untuk","pada","sebagai","dengan","atau",
    "para","serta","sebuah","seorang","sebuah","adalah","tersebut","dalam","oleh"
}

SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-žΑ-ωА-яİıĞğŞşÇçŊŋ']+")

def split_sentences(text: str):
    s = str(text).strip()
    if not s:
        return []
    # Pemisah sederhana: titik/tanya/seru/newline
    sents = [t.strip() for t in SENT_SPLIT_RE.split(s) if t.strip()]
    return sents if sents else [s]

def tokenize_words(text: str):
    return [w.lower() for w in WORD_RE.findall(str(text))]

def sent_embeddings(embedder, sents):
    if not sents:
        return np.zeros((0, 384), dtype=np.float32)
    E = embedder.encode(sents, convert_to_numpy=True, normalize_embeddings=True)
    return E

def avg_adj_cosine(embedder, sents):
    if len(sents) < 2:
        return 0.0
    E = sent_embeddings(embedder, sents)
    if len(E) < 2:
        return 0.0
    sims = (E[:-1] * E[1:]).sum(axis=1)  # cosine karena sudah dinormalisasi
    return float(np.mean(sims))

def content_overlap_ratio(s1, s2):
    tok1 = {w for w in tokenize_words(s1) if w not in ID_STOP}
    tok2 = {w for w in tokenize_words(s2) if w not in ID_STOP}
    if not tok1 or not tok2:
        return 0.0
    return len(tok1 & tok2) / len(tok1 | tok2)

def avg_content_overlap(sents):
    if len(sents) < 2:
        return 0.0
    vals = [content_overlap_ratio(a, b) for a, b in zip(sents[:-1], sents[1:])]
    return float(np.mean(vals)) if vals else 0.0

def ratio_in_vocab(tokens, vocab):
    n = len(tokens) or 1
    return sum(t in vocab for t in tokens) / n

def narrativity_score(embedder, text: str) -> float:
    sents = split_sentences(text)
    words = tokenize_words(text)
    pronoun_ratio = ratio_in_vocab(words, PRONOUNS)
    action_ratio  = ratio_in_vocab(words, ACTION_VERBS)
    temp_ratio    = ratio_in_vocab(words, TEMP_CONN)
    adj_sim       = avg_adj_cosine(embedder, sents)
    cont_overlap  = avg_content_overlap(sents)
    # Heuristik skala 0–100 (bobot bisa dikalibrasi ulang)
    score = 100 * (
        0.25 * pronoun_ratio +
        0.20 * action_ratio +
        0.10 * temp_ratio +
        0.25 * adj_sim +
        0.20 * cont_overlap
    )
    return float(np.clip(score, 0, 100))

def referential_cohesion_score(embedder, text: str) -> float:
    sents = split_sentences(text)
    adj_sim      = avg_adj_cosine(embedder, sents)
    cont_overlap = avg_content_overlap(sents)
    # Heuristik: fokus overlap & kemiripan antar kalimat
    score = 100 * (0.55 * cont_overlap + 0.45 * adj_sim)
    return float(np.clip(score, 0, 100))

def deep_cohesion_score(embedder, text: str) -> float:
    sents = split_sentences(text)
    words = tokenize_words(text)
    causal_ratio = ratio_in_vocab(words, CAUSAL_CONN)
    adj_sim      = avg_adj_cosine(embedder, sents)
    # Deep cohesion ~ eksplisitasi relasi kausal/logis + konsistensi semantik
    score = 100 * (0.55 * causal_ratio + 0.45 * adj_sim)
    return float(np.clip(score, 0, 100))

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        description="Prediksi concreteness untuk kata & (opsional) skor Narativity/Referential/Deep Cohesion untuk teks Indonesia."
    )
    ap.add_argument("--en-lexicon", default=r"versi3\lexicon\concreteness_scores_original.csv",
                    help="Path lexicon Inggris (Brysbaert).")
    ap.add_argument("--input-csv", required=True, help="CSV berisi kata dan/atau teks.")
    ap.add_argument("--output-csv", default="concreteness_id_pred.csv", help="Path output CSV.")
    ap.add_argument("--input-col", default=None,
                    help="Nama kolom kata (contoh: Kata, word, token). Jika ada, hitung concreteness.")
    ap.add_argument("--text-col", default=None,
                    help="Nama kolom teks (contoh: Teks, Abstrak, Artikel). Jika ada, hitung 3 skor teks.")
    ap.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2",
                    help="SentenceTransformer multibahasa (dipakai untuk embedding kata & kalimat).")
    ap.add_argument("--batch-size", type=int, default=256, help="Batch encode size.")
    args = ap.parse_args()

    try:
        # Muat CSV & rapikan header
        df = pd.read_csv(args.input_csv, encoding="utf-8-sig", skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]

        # Deteksi kolom kata (untuk concreteness)
        word_col = None
        if args.input_col:
            match = [c for c in df.columns if _norm(c) == _norm(args.input_col)]
            if not match:
                raise KeyError(f"Kolom '{args.input_col}' tidak ada. Kolom tersedia: {list(df.columns)}")
            word_col = match[0]
        else:
            try:
                word_col = find_col(df.columns, {"Kata","kata","word_id","word","term","token"})
            except Exception:
                word_col = None  # tidak wajib

        # Deteksi kolom teks (untuk 3 skor teks)
        text_col = None
        if args.text_col:
            match = [c for c in df.columns if _norm(c) == _norm(args.text_col)]
            if not match:
                raise KeyError(f"Kolom teks '{args.text_col}' tidak ada. Kolom tersedia: {list(df.columns)}")
            text_col = match[0]
        else:
            # coba auto-detect
            for cand in ["Teks","teks","Abstrak","abstrak","Text","text","Artikel","Isi","Kalimat","Paragraf"]:
                if cand in df.columns:
                    text_col = cand
                    break

        # Siapkan embedder & (opsional) regressor concreteness
        embedder = SentenceTransformer(args.model)

        if word_col is not None:
            # Pastikan ada regressor cache; kalau tidak, latih dari lexicon Inggris
            def ensure_conc_regressor():
                if Path(MODEL_FILE).exists() and Path(META_FILE).exists():
                    try:
                        return load_regressor()
                    except Exception:
                        pass
                lex_en = load_english_lexicon(args.en_lexicon)
                print(f"[INFO] English lexicon loaded: {len(lex_en)} entries.")
                return train_and_save_regressor(lex_en, model_name=args.model, batch_size=args.batch_size)

            # Gunakan embedder yang sama kalau cocok
            if Path(META_FILE).exists():
                try:
                    meta = json.loads(Path(META_FILE).read_text(encoding="utf-8"))
                    if meta.get("embedder") == args.model and Path(MODEL_FILE).exists():
                        _, reg = load_regressor()
                    else:
                        _, reg = ensure_conc_regressor()
                except Exception:
                    _, reg = ensure_conc_regressor()
            else:
                _, reg = ensure_conc_regressor()

            # Prediksi concreteness untuk kata unik lalu map ke semua baris
            series_words = df[word_col].astype(str).map(clean_id_word)
            uniq_words = series_words.dropna().drop_duplicates().tolist()
            print(f"[INFO] Unique ID terms to score (concreteness): {len(uniq_words)}")
            conc_preds = predict_concreteness(embedder, reg, uniq_words, batch_size=args.batch_size)
            conc_map = dict(zip(uniq_words, conc_preds))
            df["concreteness_pred"] = series_words.map(conc_map)

        # Skor berbasis teks (jika ada kolom teks)
        if text_col is not None:
            texts = df[text_col].fillna("").astype(str).tolist()
            print(f"[INFO] Rows to score (text metrics): {len(texts)}")

            narr_scores  = []
            refc_scores  = []
            deep_scores  = []
            for t in texts:
                narr_scores.append(narrativity_score(embedder, t))
                refc_scores.append(referential_cohesion_score(embedder, t))
                deep_scores.append(deep_cohesion_score(embedder, t))

            df["Narrativity"] = np.round(narr_scores, 2)
            df["Referential_Cohesion"] = np.round(refc_scores, 2)
            df["Deep_Cohesion"] = np.round(deep_scores, 2)

        # Atur urutan kolom: kata/teks & skor-skor di depan
        front_cols = []
        if word_col is not None:
            front_cols += [word_col, "concreteness_pred"]
        if text_col is not None:
            front_cols += [text_col, "Narrativity", "Referential_Cohesion", "Deep_Cohesion"]
        other_cols = [c for c in df.columns if c not in set(front_cols)]
        out_df = df[front_cols + other_cols] if front_cols else df

        out_df.to_csv(args.output_csv, index=False, encoding="utf-8")
        print(f"[OK] Saved → {args.output_csv} (rows={len(out_df)})")
        if word_col is None and text_col is None:
            print("[WARN] Tidak menemukan kolom kata maupun kolom teks. Tidak ada skor yang dihitung.")

    except Exception as e:
        print("[ERROR]", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
