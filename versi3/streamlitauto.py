# app_paraphrase_tags.py — Streamlit Paraphrase + Skoring + Filter Tag
# Path kamus: versi3/tesaurus-master/dict.json

from __future__ import annotations
import os
import json
import itertools
import re
from typing import Dict, List, Iterable, Tuple

import pandas as pd
import streamlit as st

# --- scoring core (wajib tersedia di PYTHONPATH) ---
try:
    from versi3_5 import score_text  # ganti jika modulmu bernama lain
except Exception as e:
    st.stop()  # fatal untuk aplikasi

# -----------------------------
# Konfigurasi
# -----------------------------
DEFAULT_THES_PATH = os.path.join("versi3", "tesaurus-master", "dict.json")

# Daftar tag POS
TAG_LABELS = {
    "a": "adjektiva",
    "adv": "adverbia",
    "ki": "kiasan",
    "n": "nomina",
    "num": "numeralia",
    "p": "partikel",
    "pron": "pronomina",
    "v": "verba",
}

FILLERS = {
    "yang", "dimana", "yakni", "yaitu", "adalah",
    "pada dasarnya", "dalam rangka",
}

OBJECTIVES = [
    "Flesch_Score_Normalized",   # default
    "Narrativity",
    "Syntactic_Simplicity",
    "Word_Concreteness",
    "Referential_Cohesion",
    "Deep_Cohesion",
]

WEIGHTS_DEFAULT = {
    "Narrativity": 0.20,
    "Syntactic_Simplicity": 0.20,
    "Word_Concreteness": 0.15,
    "Referential_Cohesion": 0.20,
    "Deep_Cohesion": 0.10,
    "Flesch_Score_Normalized": 0.15,
}

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\-]+")

# -----------------------------
# Loader & util
# -----------------------------
@st.cache_data(show_spinner=False)
def load_thesaurus_with_tags(path: str) -> Dict[str, Dict[str, object]]:
    """
    Return: { lemma: { 'tag': 'v', 'syns': [list[str]] } } ; semua lowercase.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, Dict[str, object]] = {}
    for lemma, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        tag = (entry.get("tag") or "").strip().lower()
        syns = entry.get("sinonim", []) or []
        syns = [s.strip().lower() for s in syns if isinstance(s, str) and s.strip()]
        lemma_l = (lemma or "").strip().lower()
        if not lemma_l or not syns:
            continue
        # unik, buang yang identik dgn lemma
        syns = sorted({s for s in syns if s != lemma_l})
        if not syns:
            continue
        out[lemma_l] = {"tag": tag, "syns": syns}
    return out

def _tok(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())

def _detok(tokens: List[str]) -> str:
    return " ".join(tokens)

def _is_allowed_by_tag(lemma: str, thes: Dict[str, Dict[str, object]], allowed_tags: set[str] | None) -> bool:
    if lemma not in thes:
        return False
    if not allowed_tags:
        return True  # All tags
    tag = str(thes[lemma].get("tag") or "").lower()
    return tag in allowed_tags

def _candidate_rewrites(
    tokens: List[str],
    thes: Dict[str, Dict[str, object]],
    allowed_tags: set[str] | None,
    max_sub_per_sentence: int,
    remove_fillers: bool,
) -> Iterable[List[str]]:
    # posisi eligible berdasarkan tag
    positions = [i for i, t in enumerate(tokens) if _is_allowed_by_tag(t, thes, allowed_tags)]
    if max_sub_per_sentence > 0:
        positions = positions[:max_sub_per_sentence]

    base = tokens

    # 1) hapus filler (opsional)
    if remove_fillers:
        no_fill = [t for t in base if t not in FILLERS]
        if no_fill and no_fill != base:
            yield no_fill

    # 2) substitusi sinonim satu posisi
    for pos in positions:
        lemma = tokens[pos]
        syns = thes.get(lemma, {}).get("syns", []) or []
        for syn in syns:
            syn_toks = _tok(syn)
            cand = list(base)
            cand[pos:pos+1] = syn_toks if syn_toks else [syn]
            yield cand

    # 3) kombinasi dua posisi (ringan)
    if len(positions) >= 2:
        for pos1, pos2 in itertools.combinations(positions, 2):
            syns1 = thes.get(tokens[pos1], {}).get("syns", []) or []
            syns2 = thes.get(tokens[pos2], {}).get("syns", []) or []
            for s1 in syns1:
                for s2 in syns2:
                    c = list(base)
                    t1 = _tok(s1); t2 = _tok(s2)
                    if pos2 > pos1:
                        c[pos2:pos2+1] = t2 if t2 else [s2]
                        c[pos1:pos1+1] = t1 if t1 else [s1]
                    else:
                        c[pos1:pos1+1] = t1 if t1 else [s1]
                        c[pos2:pos2+1] = t2 if t2 else [s2]
                    yield c

def _composite_score(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    return sum(float(scores.get(k, 0.0)) * float(w) for k, w in weights.items())

@st.cache_data(show_spinner=False, ttl=120)
def paraphrase_rank(
    text: str,
    thes: Dict[str, Dict[str, object]],
    allowed_tags: set[str] | None,
    beam_size: int = 32,
    max_sub_per_sentence: int = 3,
    remove_fillers: bool = True,
    objective: str = "Flesch_Score_Normalized",
    weights: Dict[str, float] | None = None,
    max_candidates: int = 120,
) -> pd.DataFrame:
    tokens = _tok(text)
    if not tokens:
        return pd.DataFrame(columns=["Text","Objective","Scores"])

    seen = set()
    cands: List[str] = []

    orig = _detok(tokens)
    cands.append(orig); seen.add(orig)

    for cand_tokens in _candidate_rewrites(tokens, thes, allowed_tags, max_sub_per_sentence, remove_fillers):
        t = _detok(cand_tokens)
        if t not in seen:
            seen.add(t); cands.append(t)
        if len(cands) >= max_candidates:
            break

    rows = []
    for t in cands:
        s = score_text(t)
        obj = (_composite_score(s, weights) if objective == "Weighted_Composite" else float(s.get(objective, 0.0)))
        rows.append({
            "Text": t,
            "Objective": obj,
            "Narrativity": s.get("Narrativity", 0.0),
            "Syntactic_Simplicity": s.get("Syntactic_Simplicity", 0.0),
            "Word_Concreteness": s.get("Word_Concreteness", 0.0),
            "Referential_Cohesion": s.get("Referential_Cohesion", 0.0),
            "Deep_Cohesion": s.get("Deep_Cohesion", 0.0),
            "Flesch_Score_Normalized": s.get("Flesch_Score_Normalized", 0.0),
            "Flesch_Category": s.get("Flesch_Category", ""),
            "Scores": s,
        })

    df = pd.DataFrame(rows).sort_values("Objective", ascending=False)
    return df.head(beam_size)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Paraphrase + Tag Filter (TERA-ID)", page_icon="🧠", layout="wide")
st.title("🧠 Paraphrase Sinonim (Filter Tag) + Skoring TERA-ID")

with st.sidebar:
    st.header("📂 Thesaurus")
    thes_path = st.text_input("Path dict.json", value=DEFAULT_THES_PATH, help="Contoh: versi3/tesaurus-master/dict.json")
    if not os.path.isfile(thes_path):
        st.warning("File tidak ditemukan. Periksa path.")
        st.stop()
    thes = load_thesaurus_with_tags(thes_path)
    st.caption(f"Entries: **{len(thes):,}**")

    st.header("⚙️ Pengaturan")
    objective = st.selectbox("Objective ranking", OBJECTIVES, index=0)
    use_composite = st.checkbox("Pakai Weighted Composite", value=False)
    weights = WEIGHTS_DEFAULT.copy()
    if use_composite:
        for k in list(weights.keys()):
            weights[k] = st.slider(k, 0.0, 1.0, float(weights[k]), 0.05)
        tot = sum(weights.values()) or 1.0
        weights = {k: v/tot for k, v in weights.items()}

    beam_size = st.slider("Top-N ditampilkan", 5, 100, 20, step=5)
    max_sub = st.slider("Maks. posisi sinonim", 0, 5, 3)
    max_cands = st.slider("Maks. kandidat digenerate", 5, 500, 120, step=5)
    remove_fillers = st.checkbox("Hapus kata tak efektif", value=True)

    st.header("🏷️ Filter Tag")
    chosen = st.multiselect(
        "Pilih tag (kosong = semua)",
        options=list(TAG_LABELS.keys()),
        format_func=lambda k: f"{k} = {TAG_LABELS[k]}",
    )
    allowed_tags = set(chosen) if chosen else None

sample = "Penelitian ini bertujuan untuk mengetahui pengaruh komposisi media tanam terhadap pertumbuhan vanili."
text = st.text_area("Teks sumber", value=sample, height=160)

# Tata letak vertikal (atas-bawah)

st.subheader("🔁 Generate Kandidat")
run = st.button("Parafrase & Skor")
st.caption("Filter hanya berlaku pada **lemma** yang tag-nya cocok.")

if run:
    with st.spinner("Menghasilkan kandidat & menghitung skor..."):
        df = paraphrase_rank(
            text=text,
            thes=thes,
            allowed_tags=allowed_tags,
            beam_size=beam_size,
            max_sub_per_sentence=max_sub,
            remove_fillers=remove_fillers,
            objective=("Weighted_Composite" if use_composite else objective),
            weights=(weights if use_composite else None),
            max_candidates=max_cands,
        )
    st.session_state["df"] = df
    st.success(f"✅ {len(df)} kandidat teratas siap.")

st.markdown("---")
st.subheader("🏆 Ranking Kandidat (Skor tertinggi → terendah)")
df = st.session_state.get("df")
if isinstance(df, pd.DataFrame) and not df.empty:
    show_cols = [
        "Objective","Text",
        "Flesch_Score_Normalized","Narrativity","Syntactic_Simplicity",
        "Word_Concreteness","Referential_Cohesion","Deep_Cohesion","Flesch_Category"
    ]
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)
    best = df.iloc[0]
    st.markdown("**Pilihan Terbaik:**")
    st.text_area("Teks Terbaik", value=str(best["Text"]), height=140)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Unduh kandidat.csv", data=csv_bytes, file_name="kandidat.csv", mime="text/csv")
else:
    st.info("Belum ada hasil. Jalankan *Parafrase & Skor*.")

st.sidebar.markdown("---")
st.sidebar.caption("Tag: a/adv/ki/n/num/p/pron/v. Filter kosong = semua tag.")
