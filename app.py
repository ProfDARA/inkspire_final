# app.py — Readability scorer + per-sentence paraphrase (with scores) + polished UI
import streamlit as st
import pandas as pd
import numpy as np
import joblib, torch, re, nltk, os, json
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import stanza
from transformers import AutoModelForSeq2SeqLM

# ---------------- Setup ----------------
nltk.download("punkt", quiet=True)
st.set_page_config(page_title="Keterbacaan & Parafrasa (ID)", layout="wide")

# ====== Simple theme polish ======
st.markdown("""
<style>
    .main { padding-top: 1.2rem; }
    h1, h2, h3 { font-weight: 700; }
    .metric-card { padding: 14px 16px; border-radius: 12px; border: 1px solid #EEE; background: #FAFAFA; }
    .pill { display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; border:1px solid #ddd; background:#f7f7f7; }
    .good { background:#E8F5E9; border-color:#C8E6C9; }
    .warn { background:#FFF8E1; border-color:#FFECB3; }
    .bad  { background:#FDECEA; border-color:#FADBD8; }
    .small { color:#666; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ===== Cache loaders =====
@st.cache_resource
def load_meta():
    return joblib.load("meta.joblib")

@st.cache_resource
def load_models(targets):
    return {t: joblib.load(f"model_{t}.joblib") for t in targets}

@st.cache_resource
def load_text_models(hf_model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(hf_model_name)
    bert = AutoModel.from_pretrained(hf_model_name).to(device).eval()
    sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return tok, bert, sbert, device

@st.cache_resource
def load_stanza():
    try:
        return stanza.Pipeline(
            "id", processors="tokenize,pos",
            tokenize_no_ssplit=True,
            use_gpu=torch.cuda.is_available(),
            verbose=False, tokenize_batch_size=10
        )
    except Exception:
        stanza.download("id")
        return stanza.Pipeline(
            "id", processors="tokenize,pos",
            tokenize_no_ssplit=True,
            use_gpu=torch.cuda.is_available(),
            verbose=False, tokenize_batch_size=10
        )

@st.cache_resource
def load_t5(ckpt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(ckpt, use_fast=False)
    t5  = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(device).eval()
    if device.type == "cuda":
        try: t5.half()
        except: pass
    return tok, t5, device

# ===== Utils =====
ID_STOP = set("yang dan di ke dari untuk pada dengan kepada sebagai adalah ialah yakni yaitu atau namun tetapi karena sehingga maka agar supaya sebuah para itu ini pun jika bila apabila saat ketika sementara selama hingga kemudian telah sudah akan".split())

# token regexs
ML_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ]+", re.UNICODE)  # untuk fitur konteks (stabil seperti training)
TOKEN_RE    = re.compile(r"""
    [A-Za-zÀ-ÿ]+(?:-[A-Za-zÀ-ÿ]+)*      # kata/istilah (boleh hyphen)
  | \d+(?:[.,]\d+)?(?:[A-Za-z%°µμ]+)?   # angka, desimal + unit menempel (MPa, mm, ul, %)
""", re.X)

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
INSTR_RE   = re.compile(r"\b(tulis|ubah|kalimat|istilah|teknis|instruksi|input|output|teks|parafrasa)\b", re.I)
VERB_HINT  = re.compile(r"\b(di|ke|meng|meny|mem|ter|ber|se)\w+", re.I)

def norm_tokens(s: str):
    toks = [t.lower() for t in ML_TOKEN_RE.findall(str(s))]
    toks = [t for t in toks if t not in ID_STOP and not t.isnumeric() and len(t) > 2]
    return toks

def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb: return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def mean_pooling(h, mask):
    mask = mask.unsqueeze(-1).expand(h.size()).float()
    return (h*mask).sum(1) / mask.sum(1).clamp(min=1e-9)

@torch.no_grad()
def embed_indobert(texts, tok, bert, device, max_len=128, bs=128):
    outs=[]
    for i in range(0,len(texts),bs):
        batch = texts[i:i+bs]
        enc = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_len).to(device)
        h = bert(**enc).last_hidden_state
        pooled = mean_pooling(h, enc["attention_mask"]).cpu().numpy()
        outs.append(pooled)
    return np.vstack(outs)

def build_context_features(texts, sbert):
    embs = sbert.encode(texts, batch_size=512, convert_to_numpy=True, normalize_embeddings=True)
    N = len(texts)
    sim_prev = np.zeros(N); sim_next = np.zeros(N)
    overlap_prev = np.zeros(N); overlap_next = np.zeros(N)
    toks = [norm_tokens(t) for t in texts]
    for i in range(N):
        if i-1 >= 0:
            sim_prev[i] = float(np.dot(embs[i], embs[i-1]))
            overlap_prev[i] = jaccard(toks[i], toks[i-1])
        if i+1 < N:
            sim_next[i] = float(np.dot(embs[i], embs[i+1]))
            overlap_next[i] = jaccard(toks[i], toks[i+1])
    return np.column_stack([sim_prev, sim_next, overlap_prev, overlap_next])

def build_pos_features(texts, pos_tags, stz_pipeline, batch_size=10):
    if not pos_tags:
        return np.zeros((len(texts), 0), dtype=np.float32)
    feats = np.zeros((len(texts), len(pos_tags)), dtype=np.float32)
    tag_index = {t: i for i, t in enumerate(pos_tags)}
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start+batch_size]
        docs = [stz_pipeline(str(txt)) for txt in batch]
        for j, doc in enumerate(docs):
            counts = np.zeros(len(pos_tags), dtype=np.float32)
            total = 0
            for sent in doc.sentences:
                for w in sent.words:
                    total += 1
                    if w.upos in tag_index:
                        counts[tag_index[w.upos]] += 1
            if total > 0:
                counts /= total
            feats[start+j, :] = counts
    return feats

def make_feature_matrix(df, text_col, tok, bert, sbert, device, meta, stz_pipeline):
    texts = df[text_col].fillna("").astype(str).tolist()
    # 1) IndoBERT
    X_bert = embed_indobert(texts, tok, bert, device)
    # 2) Context (SBERT)
    X_ctx = build_context_features(texts, sbert)
    # 3) POS (kalau dipakai saat training)
    X_pos = np.zeros((len(texts), 0), dtype=np.float32)
    if meta.get("use_pos_features", False):
        pos_tags = meta.get("pos_tags", [])
        X_pos = build_pos_features(texts, pos_tags, stz_pipeline, batch_size=10)
    # Urutan fitur WAJIB: [BERT, Context, POS]
    return np.hstack([X_bert, X_ctx, X_pos])

def categorize(val):
    if val >= 80: return "Sangat Mudah"
    if val >= 60: return "Mudah"
    if val >= 40: return "Sedang"
    if val >= 20: return "Sulit"
    return "Sangat Sulit"

# ===== Paraphrase helpers (adapted to accept keep_sentence flag) =====
def make_bad_words_ids(tok):
    vocab = tok.get_vocab()
    ban = [
        # instruksi umum
        "Tulis","tulis","Ubah","ubah","kalimat","Kalimat","aktif","Aktif",
        "ilmiah","Instruksi","instruksi","Pertahankan","istilah","teknis",
        "Parafrasa","parafrasa","Input","Output","output","Teks","teks",
        "Constraints","constraints","Keep","KEEP","|||","<R>","<<",">>", "`","``","''",
        # tag gaya
        "Sedang","Mudah","Sulit","F_Sedang","F_Mudah","F_Sulit","<F_Sedang>","<F_Mudah>","<F_Sulit>",
    ]
    ids = [[vocab[w]] for w in ban if w in vocab]
    for i in range(200):
        t = f"<extra_id_{i}>"
        if t in vocab: ids.append([vocab[t]])
    return ids

def clean_once(s: str) -> str:
    s = re.sub(r"<extra_id_\d+>", "", s)
    s = re.sub(r"(paraphrase:|Constraints:|Keep:\[.*?\]\s*|\|\|\|).*", "", s, flags=re.I)
    s = INSTR_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def hard_cap_words(s: str, cap: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    toks = TOKEN_RE.findall(s)
    if len(toks) > cap:
        s = " ".join(toks[:cap])
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s and s[-1] not in ".!?":
        s += "."
    return s

def too_similar(a: str, b: str, thr=0.92) -> bool:
    A = set(" ".join(TOKEN_RE.findall(a.lower())).split())
    B = set(" ".join(TOKEN_RE.findall(b.lower())).split())
    if not A or not B: return False
    return (len(A & B) / max(1, len(A | B))) >= thr

def build_prompt(src: str, keep_terms=None, cap: int = 22, keep_sentence: bool = False) -> str:
    keep = ", ".join(keep_terms or [])
    base = (
        f"paraphrase: {src} ||| "
        f"Constraints: satu kalimat Indonesia yang ringkas (≤{cap} kata), kalimat aktif, tetap ilmiah. "
        f"Keep:[{keep}]"
    )
    if keep_sentence:
        # Instruksi tambahan untuk mempertahankan sebagian struktur asli
        base += " Pertahankan sebagian struktur dan makna kalimat asli."
    return base

@torch.no_grad()
def generate_paraphrase(sent: str, keep_terms, t5_tok, t5_model, t5_device, cap=22, bad_ids=None, keep_sentence=False):
    prompt = build_prompt(sent, keep_terms, cap=cap, keep_sentence=keep_sentence)
    enc = t5_tok([prompt], return_tensors="pt", truncation=True, max_length=448).to(t5_device)

    # deterministik dulu
    out = t5_model.generate(
        **enc,
        max_new_tokens=min(96, cap + 12),
        num_beams=4,
        do_sample=False,
        no_repeat_ngram_size=4,
        repetition_penalty=1.18,
        bad_words_ids=bad_ids,
        use_cache=True,
    )
    txt = t5_tok.batch_decode(out, skip_special_tokens=True)[0]
    cand = hard_cap_words(clean_once(txt), cap)

    # fallback sampling jika terlalu mirip/pendek
    if len(TOKEN_RE.findall(cand)) < 8 or too_similar(sent, cand):
        out = t5_model.generate(
            **enc,
            max_new_tokens=min(96, cap + 12),
            num_beams=1,
            do_sample=True,
            top_p=0.92,
            top_k=50,
            temperature=0.8,
            no_repeat_ngram_size=4,
            repetition_penalty=1.18,
            bad_words_ids=bad_ids,
            use_cache=True,
        )
        txt = t5_tok.batch_decode(out, skip_special_tokens=True)[0]
        cand = hard_cap_words(clean_once(txt), cap)

    # fallback kalau kosong/terlalu pendek → kembalikan kalimat asli
    if not cand or not cand.strip() or len(TOKEN_RE.findall(cand)) < 6:
        cand = sent.strip()
    return cand

# ================== UI ==================
st.title("Program Skoring Keterbacaan dan Parafrase Bahasa Indonesia")
st.caption("Pipeline : IndoBERT (teks) + SBERT (konteks) + (opsional) POS → model keterbacaan. Parafrasa T5 opsional.")
st.caption("Status Proyek : Eksperimental / Dalam Pengebangan")

meta = load_meta()
hf_model = meta["hf_model"]
text_col = meta["text_col"]
targets = meta["targets"]

models = load_models(targets)
tok, bert, sbert, device = load_text_models(hf_model)
stz_pipeline = load_stanza()

with st.sidebar:
    st.header("⚙️ Paraphraser (opsional)")
    use_paraphrase = st.checkbox("Aktifkan paraphrase T5", value=True)
    ckpt_path = st.text_input("Checkpoint T5", value="./indoT5-readability-lora-v2")
    max_words = st.slider("Batas kata output (parafrasa)", 16, 40, 30, 1)
    st.markdown("### 🧙‍ Pertahankan Kalimat Aneh-Aneh")
    keep_sentence_flag = st.checkbox("Pertahankan sebagian struktur kalimat asli", value=False)
    st.markdown("_Prosesnya ajaib, ga usah mikir, cukup centang, klik proses dan lihat hasilnya!_")
    if use_paraphrase:
        try:
            t5_tok, t5_model, t5_dev = load_t5(ckpt_path)
            BAD = make_bad_words_ids(t5_tok)
            st.caption("T5 siap ✅")
        except Exception as e:
            use_paraphrase = False
            st.error(f"Gagal load T5: {e}")

st.subheader("Input Abstrak Penuh")
abs_text = st.text_area("Input abstrak di sini", "Mahasiswa sering menganggap skripsi sebagai tugas yang paling berat dan sulit karena harus melalui proses penelitian yang panjang dan membutuhkan usaha yang lebih besar daripada saat menyelesaikan tugas perkuliahan. Tidak jarang terdapat mahasiswa yang mengalami tekanan karena skripsi. Perasaan terbebani dan tidak mampu dalam proses pengerjaan skripsi sering kali menyebabkan mahasiswa merasakan stres akademik. Stres akademik yang dihadapi mahasiswa mampu memicu munculnya berbagai gangguan psikologis, salah satunya yaitu eating disorder atau gangguan makan. Penelitian ini ditujukan guna menganalisis korelasi antara stres akademik dengan kecenderungan gangguan pola makan pada mahasiswa Program Studi Pendidikan Bahasa Inggris di Universitas Negeri Yogyakarta yang sedang menyelesaikan skripsi. Penelitian ini memakai metode penelitian kuantitatif, serta teknik analisis data korelasi pearson. Hasil penelitian menunjukkan bahwasanya ada korelasi positif dan signifikan antara stress akademik dengan kecenderungan eating disorder pada mahasiswa Program Studi Pendidikan Bahasa Inggris yang sedang menyusun skripsi di Universitas Negeri Yogyakarta (0,000 0,005). Semakin tinggi tingkat stress akademik yang dialami, semakin tinggi juga mahasiswa memiliki kecenderungan untuk mengalami eating disorder.", height=160)

if st.button("Proses"):
    # 1) Split per kalimat sederhana
    sents = [s.strip() for s in SENT_SPLIT.split(abs_text.strip()) if s.strip()]
    if not sents:
        st.warning("Tidak ada kalimat terdeteksi.")
        st.stop()

    # 2) Siapkan DF input
    df_in = pd.DataFrame({text_col: sents})

    # 3) Fitur & prediksi untuk kalimat asli
    X_orig = make_feature_matrix(df_in, text_col, tok, bert, sbert, device, meta, stz_pipeline)
    for tgt in targets:
        df_in[f"old_{tgt}"] = models[tgt].predict(X_orig)
    df_in["Kategori (Asli)"] = df_in["old_Flesch_Normalized"].apply(categorize)

    # 4) (Opsional) Parafrasa + skor
    if use_paraphrase:
        paras = []
        for s in sents:
            # keep terms kecil: ambil kata kapital/istilah panjang dari s
            terms = []
            for w in TOKEN_RE.findall(s):
                if len(w) >= 8 or re.match(r"^[A-Z][a-zA-Z\-]+$", w):
                    terms.append(w)
            terms = list(dict.fromkeys(terms))[:5]
            para = generate_paraphrase(s, terms, t5_tok, t5_model, t5_dev, cap=max_words, bad_ids=BAD, keep_sentence=keep_sentence_flag)
            if not para.strip():  # jaga-jaga
                para = s.strip()
            paras.append(para)

        df_par = pd.DataFrame({text_col: paras})
        X_par = make_feature_matrix(df_par, text_col, tok, bert, sbert, device, meta, stz_pipeline)
        for tgt in targets:
            df_in[f"new_{tgt}"] = models[tgt].predict(X_par)
            df_in[f"gain_{tgt}"] = df_in[f"new_{tgt}"] - df_in[f"old_{tgt}"]
        df_in["Kategori (Parafrasa)"] = df_in["new_Flesch_Normalized"].apply(categorize)
        df_in["Parafrasa"] = paras

    # 5) Ringkasan Agregat — tampilkan semua skor dan delta
    st.markdown("### 📊 Ringkasan Agregat")

    agg_orig = {t: df_in[f"old_{t}"].mean() for t in targets}
    if use_paraphrase:
        agg_new = {t: df_in[f"new_{t}"].mean() for t in targets}
        deltas = {t: agg_new[t] - agg_orig[t] for t in targets}

    cols = st.columns(len(targets))
    for i, tgt in enumerate(targets):
        with cols[i]:
            if use_paraphrase:
                st.metric(
                    label=f"{tgt}",
                    value=f"{agg_new[tgt]:.2f}",
                    delta=f"{deltas[tgt]:+.2f}"
                )
                st.markdown(
                    f"<span class='small'>Asli: {agg_orig[tgt]:.2f} — "
                    f"Kategori baru: <b>{categorize(agg_new[tgt])}</b></span>",
                    unsafe_allow_html=True,
                )
            else:
                st.metric(label=f"{tgt} (Asli)", value=f"{agg_orig[tgt]:.2f}")
                st.markdown(
                    f"<span class='small'>Kategori: <b>{categorize(agg_orig[tgt])}</b></span>",
                    unsafe_allow_html=True,
                )

    # 6) Tabel per kalimat
    st.markdown("### 🧾 Hasil per Kalimat")
    show_cols = [text_col]
    if use_paraphrase: show_cols += ["Parafrasa"]
    show_cols += [f"old_{t}" for t in targets]
    if use_paraphrase:
        show_cols += [f"new_{t}" for t in targets]
        show_cols += [f"gain_{t}" for t in targets]
        show_cols += ["Kategori (Asli)", "Kategori (Parafrasa)"]
    else:
        show_cols += ["Kategori (Asli)"]

    pretty = {
        text_col: "Kalimat",
        **{f"old_{t}": f"{t} (Asli)" for t in targets},
        **{f"new_{t}": f"{t} (Parafrasa)" for t in targets},
        **{f"gain_{t}": f"Δ {t}" for t in targets},
    }

    st.dataframe(df_in[show_cols].rename(columns=pretty), use_container_width=True)

    # 7) Unduh CSV
    csv_bytes = df_in[show_cols].rename(columns=pretty).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Unduh CSV",
        data=csv_bytes,
        file_name="laporan_keterbacaan_per_kalimat.csv",
        mime="text/csv"
    )

    # 8) Gabungan paragraf — hanya versi parafrasa
    st.markdown("---")
    st.subheader("📝 Gabungan Paragraf Abstrak (Siap Review & Copy)")

    if use_paraphrase:
        seq = df_in.get("Parafrasa", pd.Series([], dtype=str)).fillna("").astype(str).tolist()
        seq = [a if a.strip() else b for a, b in zip(seq, df_in[text_col].astype(str).tolist())]
    else:
        seq = df_in[text_col].fillna("").astype(str).tolist()

    joined = " ".join(s.strip() for s in seq if s.strip())
    joined = re.sub(r"\s+([,.;:!?])", r"\1", joined)
    joined = re.sub(r"\s+", " ", joined).strip()
    if joined and joined[-1] not in ".!?":
        joined += "."

    st.text_area("Paragraf hasil gabungan", joined, height=220, key="combined_paragraph_output")

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "💾 Download .TXT",
            data=joined.encode("utf-8"),
            file_name="abstrak_parafrasa.txt",
            mime="text/plain",
        )
    with col_b:
        import streamlit.components.v1 as components
        btn_id = "copy_btn_abstract"
        components.html(
            f"""
            <button id="{btn_id}" style="
                padding:10px 16px;border-radius:10px;border:1px solid #e2e2e2;cursor:pointer;
                background:#f9f9f9;">📋 Copy ke Clipboard</button>
            <script>
              const btn = document.getElementById("{btn_id}");
              if (btn) {{
                btn.onclick = async () => {{
                  try {{
                    await navigator.clipboard.writeText({json.dumps(joined)});
                    btn.innerText = "✅ Tersalin!";
                    setTimeout(() => btn.innerText = "📋 Copy ke Clipboard", 1500);
                  }} catch (e) {{
                    alert("Gagal menyalin. Coba manual Ctrl/Cmd+C.");
                  }}
                }};
              }}
            </script>
            """,
            height=50,
        )
