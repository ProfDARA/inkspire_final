# versi7/inference_clean.py
# -*- coding: utf-8 -*-
import re, torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------- Config ----------
CKPT = r"./indoT5-readability-lora-v2"  # path model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Load ----------
tok = AutoTokenizer.from_pretrained(CKPT, use_fast=False)
t5  = AutoModelForSeq2SeqLM.from_pretrained(CKPT).to(DEVICE).eval()
if DEVICE == "cuda":
    try: t5.half()
    except: pass

# ---------- Regex & utils ----------
WORD_RE   = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-žΑ-ωА-я'’\-]+")
INSTR_RE  = re.compile(r"\b(tulis|ubah|kalimat|istilah|teknis|instruksi|input|output|teks|parafrasa)\b", re.I)
SPLIT_RE  = re.compile(r"(?<=[\.\!\?])\s+")
VERB_HINT = re.compile(r"\b(di|ke|meng|meny|mem|ter|ber|se)\w+", re.I)

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def hard_cap_words(s: str, cap: int) -> str:
    s = normalize_spaces(s)
    toks = WORD_RE.findall(s)
    if len(toks) > cap:
        s = " ".join(toks[:cap])
    s = s.rstrip(" ,;:—–-")
    if s and s[-1] not in ".!?":
        s += "."
    return s

_TINY_TAIL = re.compile(r"\b([a-zA-Z]{1,4})\.$")
def heal_tail(s: str, src: str) -> str:
    s = normalize_spaces(s)
    m = _TINY_TAIL.search(s)
    if not m:
        return s
    frag = m.group(1)
    if len(frag) <= 3:
        s = s[:m.start()].rstrip(" ,;:—–-") + "."
        return s
    src_words = [w for w in WORD_RE.findall(src) if len(w) >= len(frag)+2]
    cand = next((w for w in src_words if w.lower().startswith(frag.lower())), None)
    if cand:
        s = s[:m.start()] + cand + "."
    else:
        s = s[:m.start()].rstrip(" ,;:—–-") + "."
    return s

def too_similar(a: str, b: str, thr=0.92) -> bool:
    A = set(" ".join(WORD_RE.findall(a.lower())).split())
    B = set(" ".join(WORD_RE.findall(b.lower())).split())
    if not A or not B: return False
    return (len(A & B) / max(1, len(A | B))) >= thr

# ---------- Blacklist token ----------
def make_bad_words_ids(tok):
    vocab = tok.get_vocab()
    ban = [
        "Tulis","tulis","Ubah","ubah","kalimat","Kalimat","aktif","Aktif",
        "ilmiah","Instruksi","instruksi","Pertahankan","istilah","teknis",
        "Parafrasa","parafrasa","Input","Output","output","Teks","teks",
        "Constraints","constraints","Keep","KEEP","|||","<R>","<<",">>", "`","``","''",
        "Sedang","Mudah","Sulit","F_Sedang","F_Mudah","F_Sulit","<F_Sedang>","<F_Mudah>","<F_Sulit>"
    ]
    ids = [[vocab[w]] for w in ban if w in vocab]
    for i in range(200):  # sentinel T5
        t = f"<extra_id_{i}>"
        if t in vocab: ids.append([vocab[t]])
    return ids

BAD = make_bad_words_ids(tok)

# ---------- Prompt (match training) ----------
def build_prompt(src: str, keep_terms=None, cap: int = 22, keep_sentence: bool = False) -> str:
    keep = ", ".join(keep_terms or [])
    base = (
        f"paraphrase: {src} ||| "
        f"Constraints: satu kalimat Indonesia yang ringkas (≤{cap} kata), kalimat aktif, tetap ilmiah. "
        f"Keep:[{keep}]"
    )
    if keep_sentence:
        base += " Pertahankan sebagian struktur dan makna kalimat asli."
    return base

# ---------- Cleaning ----------
def clean_once(s: str) -> str:
    s = re.sub(r"<extra_id_\d+>", "", s)
    s = re.sub(r"(paraphrase:|Constraints:|Keep:\[.*?\]\s*|\|\|\|).*", "", s, flags=re.I)
    s = INSTR_RE.sub(" ", s)
    s = normalize_spaces(s)
    return s

# ---------- Chunking lembut ----------
CLAUSE_SPLIT = re.compile(r"(?:(?<=,)|;|:|—|–|-|\(|\))\s+")
SOFT_CONJ    = re.compile(r"\b(yang|bahwa|sehingga|agar|ketika|sementara|apabila|walaupun|meskipun|serta|dan|atau|dengan|untuk)\b", re.I)

def split_long_sentence(s: str) -> List[str]:
    chunks = [p.strip() for p in CLAUSE_SPLIT.split(s) if p.strip()]
    out = []
    for p in chunks:
        if len(WORD_RE.findall(p)) >= 40:
            parts = SOFT_CONJ.split(p)
            merged = []
            i = 0
            while i < len(parts):
                if i+1 < len(parts):
                    merged.append((parts[i] + " " + parts[i+1]).strip())
                    i += 2
                else:
                    merged.append(parts[i].strip())
                    i += 1
            out += [m for m in merged if m]
        else:
            out.append(p)
    return out[:6]

# ---------- Generator sekali + retry bila tail terpotong ----------
@torch.no_grad()
def generate_once(text: str, keep_terms=None, cap: int = 22, beam=2, sample=False, keep_sentence=False):
    prompt = build_prompt(text, keep_terms, cap=cap, keep_sentence=keep_sentence)
    enc = tok([prompt], return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    out = t5.generate(
        **enc,
        max_new_tokens=min(128, cap + 20),
        min_new_tokens=10,
        num_beams=beam,
        do_sample=bool(sample),
        top_p=0.92 if sample else None,
        top_k=50 if sample else None,
        temperature=0.85 if sample else None,
        no_repeat_ngram_size=4,
        repetition_penalty=1.15,
        bad_words_ids=BAD,
        use_cache=True,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    txt = tok.batch_decode(out, skip_special_tokens=True)[0]
    txt = heal_tail(hard_cap_words(clean_once(txt), cap), text)

    if _TINY_TAIL.search(txt):
        out = t5.generate(
            **enc,
            max_new_tokens=min(128, cap + 20),
            min_new_tokens=10,
            num_beams=1,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            temperature=0.9,
            no_repeat_ngram_size=4,
            repetition_penalty=1.1,
            bad_words_ids=BAD,
            use_cache=True,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        txt2 = heal_tail(hard_cap_words(clean_once(tok.batch_decode(out, skip_special_tokens=True)[0]), cap), text)
        if (not _TINY_TAIL.search(txt2)) or (len(WORD_RE.findall(txt2)) >= len(WORD_RE.findall(txt))):
            txt = txt2
    return txt

# ---------- Paraphrase utama (panjang-aware + adaptif) ----------
@torch.no_grad()
def paraphrase(sent: str, keep_terms=None, keep_sentence=False) -> str:
    n_in = len(WORD_RE.findall(sent))
    cap  = 22 if n_in <= 30 else min(48, max(26, int(n_in * 0.75)))

    # pendek → langsung
    if n_in <= 40:
        cand = generate_once(sent, keep_terms, cap=cap, beam=2, sample=False, keep_sentence=keep_sentence)
        if len(WORD_RE.findall(cand)) < 9 or too_similar(sent, cand):
            cand = generate_once(sent, keep_terms, cap=cap, beam=1, sample=True, keep_sentence=keep_sentence)
        return cand

    # panjang → chunking + stitch
    clauses = split_long_sentence(sent)
    outs = []
    for cl in clauses:
        c = generate_once(cl, keep_terms, cap=max(18, min(24, cap-2)), beam=1, sample=False, keep_sentence=keep_sentence)
        if len(WORD_RE.findall(c)) < 7:
            c = generate_once(cl, keep_terms, cap=max(18, min(24, cap-2)), beam=1, sample=True, keep_sentence=keep_sentence)
        outs.append(c.rstrip("."))
    stitched = ", ".join([o for o in outs if o])
    stitched = normalize_spaces(stitched.replace(";", ","))
    stitched = heal_tail(hard_cap_words(stitched, cap), sent)

    if too_similar(sent, stitched) or len(WORD_RE.findall(stitched)) < 12:
        stitched = generate_once(sent, keep_terms, cap=cap, beam=1, sample=True, keep_sentence=keep_sentence)
    return stitched

# ---------- Demo usage ----------
if __name__ == "__main__":
    s = ("Metode yang digunakan adalah kualitatif deskriptif dengan pendekatan studi kasus "
         "yang mana data dikumpulkan melalui wawancara observasi dan dokumentasi yang kemudian "
         "dianalisis menggunakan teknik analisis data Miles dan Huberman namun tidak dijelaskan secara rinci")
    keep = ["wawancara", "kualitatif deskriptif"]
    out = paraphrase(s, keep_terms=keep, keep_sentence=True)
    print("Input :", s)
    print("Output:", out)
