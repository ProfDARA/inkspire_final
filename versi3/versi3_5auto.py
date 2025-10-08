"""
Mini Pipeline: TERA-style Readability Metrics for Indonesian (0–100)

TIDAR
Text Insight for Discourse Accessibility and Readability

Metrics produced per text:
- Narrativity (0–100)
- Syntactic_Simplicity (0–100)
- Word_Concreteness (0–100)
- Referential_Cohesion (0–100)
- Deep_Cohesion (0–100)
- Flesch_Score_Raw (heuristic, 0–100)
- Flesch_Score_Normalized (0–100)
- Flesch_Category (Mudah/Sedang/Sulit)

Design goals:
- Dependency-light, pure Python (no heavy NLP models required)
- Heuristic but stable (bounded 0–100, no exploding values)
- Easily extensible: you can plug in IndoBERT/SBERT later to improve cohesion/narrativity

Usage example:
>>> from versi3_5 import score_text
>>> text = "Penelitian bertujuan untuk mengetahui pengaruh komposisi media tanam dan pemberian konsentrasi MSG terhadap pertumbuhan vanili (Vanilla planifolia) stek satu ruas."
>>> score_text(text)

CLI:
$ python versi3_5.py "Teks Anda di sini"
$ python versi3_5.py --csv input.csv --text-col Teks --out scores.csv
$ python versi3_5.py --stanza-dir models/stanza_resources --sbert-dir models/sbert

CSV mode expects a column with raw text.
"""
## pip install stanza sentence-transformers

from __future__ import annotations
import re
import math
import csv
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

# === Local model directories (edit to your layout) ===
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Default: models/stanza_resources and models/sbert (relative to this file)
STANZA_DIR = os.path.join(BASE_DIR, 'models', 'stanza_resources')
SBERT_DIR  = os.path.join(BASE_DIR, 'models', 'sbert')
# Allow overrides via environment variables if provided
STANZA_DIR = os.environ.get('STANZA_RESOURCES_DIR', STANZA_DIR)
SBERT_DIR  = os.environ.get('SBERT_CACHE_DIR', SBERT_DIR)

# ----------------------------
# Optional advanced integrations (auto-detect)
# ----------------------------
USE_ADVANCED = True  # set False to force simple heuristics

_stanza = None
_stanza_nlp = None
_sbert = None
_sbert_model = None

if USE_ADVANCED:
    # ---- Stanza (LOCAL ONLY; no download here) ----
    try:
        import stanza  # type: ignore
        _stanza = stanza
        os.makedirs(STANZA_DIR, exist_ok=True)
        _stanza_nlp = stanza.Pipeline(
            'id',
            processors='tokenize,pos,lemma,depparse',
            tokenize_no_ssplit=False,
            verbose=False,
            dir=STANZA_DIR,  # <= load from local resources
        )
    except Exception:
        _stanza = None
        _stanza_nlp = None

    # ---- Sentence-Transformers (prefer local cache/model) ----
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _sbert = SentenceTransformer
        local_sbert_path = os.path.join(SBERT_DIR, 'paraphrase-multilingual-MiniLM-L12-v2')
        if os.path.isdir(local_sbert_path):
            _sbert_model = SentenceTransformer(local_sbert_path)
        else:
            os.makedirs(SBERT_DIR, exist_ok=True)
            _sbert_model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2',
                cache_folder=SBERT_DIR
            )
    except Exception:
        _sbert = None
        _sbert_model = None

# ----------------------------
# Basic text utilities
# ----------------------------
VOWELS = set(list("aiueoAIUEO"))

# Indonesian stopword list (compact; extend as needed)
STOPWORDS = set(
    "yang di ke dari kepada untuk dengan pada oleh demi agar supaya sehingga karena namun tetapi sedangkan atau dan serta pun bila jika ketika maka lalu kemudian sehingga yaitu yakni adalah ialah seperti sebagai dalam terhadap antara tanpa bukan bukanlah bukanlah pun ini itu itu pun tersebut para para nya nya pun saya aku kami kita kamu Anda anda beliau dia mereka sini sana situ itu pun hal bhw bahwa hingga hingga pun tiap setiap tiapnya seluruh seluruhnya beberapa berbagai pun akan telah telahlah telahpun sudah sudahpun telah sudah pernah sedang sedanglah sedangpun telahlah belum belumlah belumpun masih masihlah masihpun akan punlah dapat bisa mungkin tentu harus wajib tak tidak bukan bukanlah jangan ya pun punlah punnya walau walaupun meski meskipun sebaliknya sekalipun selainnya selain selainpun hingga ketika saat waktu sewaktu seraya sambil sambilpun sementara sementara itu kemudian lantas lalu selanjutnya berikutnya selaku selagi selama sehabis seusai setelah sebelum sebelumya sebelum pun sesudah sesudahnya maka yaitu yakni sebab karenanya akibatnya oleh karena itu contohnya misalnya misal".split()
)

# Connectives for deep cohesion (causal/temporal/contrastive)
CONNECTIVES = {
    # causal
        "karena", "sebab", "akibatnya", "sehingga", "maka",
        "oleh", "oleh karena itu", "oleh sebab itu", "disebabkan oleh",
        "yang menyebabkan", "yang mengakibatkan", "dengan demikian",
        "sebagai akibat", "konsekuensinya", "sebagai konsekuensi", "dampaknya", "implikasinya", "berkat",

    # contrast
        "namun", "tetapi", "meski", "meskipun", "walau",
        "sebaliknya", "padahal", "berbeda dengan", "di sisi lain",
        "kendati demikian", "walaupun demikian", "akan tetapi",
        "namun demikian", "meskipun demikian", "sekalipun demikian",
        "bertolak belakang dengan",
    
    # temporal
        "kemudian", "lalu", "selanjutnya", "berikutnya",
        "sebelumnya", "setelah", "sebelum", "setelah itu",
        "pada tahap berikutnya", "pada akhirnya", "selama",
        "ketika", "saat", "seiring dengan", "hingga",
        "sementara itu", "pada saat yang sama", "di kemudian hari",
        "sepanjang", "sejak itu", "di awal", "di akhir",
    
    # exemplify
        "misal", "misalnya", "contohnya", "sebagai contoh",
        "yakni", "yaitu", "dengan kata lain", "antara lain",
        "seperti", "termasuk", "sebagai ilustrasi", "contoh kasus",
        "khususnya",
    
    # additive
        "selain itu", "di samping itu", "lebih lanjut",
        "lebih jauh", "lagipula",
    
    # conclusion"
        "sebagai kesimpulan", "kesimpulannya",
        "dapat disimpulkan bahwa", "ringkasnya", "singkatnya",
        "dari paparan tersebut",
    
    # conditional
        "jika", "apabila", "bila", "seandainya",
        "asalkan", "jikalau", "andaikan", "kecuali jika"
    
    # purpose
        "agar", "supaya", "guna", "dalam rangka",

    # comparison
        "dibandingkan dengan", "dibanding dengan",
        "seperti halnya", "serupa dengan", "berkebalikan dengan", 
    
    # definition
        "didefinisikan sebagai", "yang dimaksud dengan",
        "dalam konteks ini", "dalam hal ini",

    # attribution
        "berdasarkan", "menurut", "mengacu pada", "merujuk pada",
    
    # limitation
        "terbatas pada", "hanya saja", "sebatas"
}


# Verb/action cues for narrativity (extendable)
ACTION_VERBS = {
    # Gerak dasar
    "pergi","datang","pulang","berjalan","berlari","melompat","masuk","keluar","naik","turun",
    "mendekat","menjauh","kembali","berputar","berhenti","bergerak",
    # Interaksi / komunikasi
    "berkata","bercerita","menceritakan","mengisahkan","menjelaskan","menjawab","bertanya",
    "mengungkapkan","mengucapkan","menyebutkan","memberitahu","menyampaikan","menegaskan",
    "menambahkan","mengulangi","mendengar","menyimak","menanggapi","membantah",
    # Persepsi / kognisi
    "melihat","menatap","memandang","memperhatikan","mengamati","menemukan","menyadari",
    "mengingat","membayangkan","memikirkan","menganalisis","menilai","mengukur","menghitung",
    "mengidentifikasi","mencatat","menafsirkan","mengecek","memeriksa","meneliti","mengumpulkan",
    "menyimpulkan","membandingkan","memutuskan","memilih","merencanakan","mengembangkan",
    # Tindakan fisik / manipulasi objek
    "mengambil","mengerjakan","menaruh","meletakkan","membawa","memindahkan","membuka","menutup","memasukkan",
    "mengeluarkan","menyusun","menyiapkan","membuat","membangun","mencetak","memotong",
    "mengaduk","mencampur","memasak","menggoreng","memanggang","mencuci","membersihkan",
    "mengeringkan","mengganti","memperbaiki","memasang","melepas","mengikat","menggosok",
    "menekan","menarik","mendorong","menggenggam","menyalakan","mematikan","menghidupkan",
    # Produksi / dokumen / teknologi
    "menulis","menyalin","mengedit","menghapus","mengunggah","mengunduh","menyimpan",
    "mengirim","mengirimkan","mengunduh","memprogram","mengkode","menjalankan","mengoperasikan",
    # Proses / perubahan
    "terjadi","muncul","timbul","membesar","mengecil","berubah","menghilang","membeku",
    "mencair","mendadak","membentuk","menghasilkan","mengakibatkan","menyebabkan",
    # Sosial / bantuan / konflik
    "membantu","menolong","menyelamatkan","menjaga","melindungi","mengawal","menemani",
    "menghadapi","menyerang","menolak","menerima","menegur","memarahi","menghibur",
    "menghadiri","mengikuti","berkumpul","bertemu","berdiskusi","berkolaborasi",
    # Emosi / reaksi
    "tersenyum","tertawa","menangis","meratap","mengeluh","menghela","menarik napas",
    "terkejut","kaget","marah","kesal","bangga","gugup","cemas","tenang","lega","sedih",
    # Hasil / capaian
    "mencoba","berhasil","gagal","mencapai","meraih","mendapatkan","kehilangan","menemukan",
    "mengumpulkan","menyelesaikan","melanjutkan","menghentikan","mengakhiri","memulai",
    # Pencarian / orientasi
    "mencari","menemukan","menelusuri","mengejar","melacak","mengikuti","mengawasi",
    # Waktu / rutinitas
    "menunggu","menjadwalkan","menunda","mempercepat","melanjutkan","mengulang","mengawali",
    # Fisiologis
    "makan","memakan","minum","meminum","tidur","bangun","bernapas","beristirahat",
    # Lain naratif umum
    "berjuang","bertahan","menyerah","menyatu","memisahkan","menghubungkan","menyatukan",
    "mengendalikan","menguasai","mengelola","mengurangi","menambah","meningkatkan",
    "menurunkan","mempercepat","memperlambat","menyaring","mengurutkan","menyalurkan"
}
# ===== Narativity feature extractors (ID) =====

TIME_WORDS = {
    "kemarin","tadi","besok","nanti","sekarang","pada","pukul","jam",
    "minggu","bulan","tahun","januari","februari","maret","april","mei","juni",
    "juli","agustus","september","oktober","november","desember","hari","senin",
    "selasa","rabu","kamis","jumat","sabtu","minggu"
}
PLACE_PREPS = {"di","ke","dari"}
PASSIVE_HINTS = {"oleh"}  # + prefix 'di-' di kata kerja

def _has_time_place(text: str, toks: List[str]) -> Tuple[float, float]:
    tset = set(toks)
    time_hit = any(w in tset for w in TIME_WORDS) or bool(re.search(r"\b(20\d{2}|19\d{2})\b", text))
    # tempat: preposisi tempat + kata setelahnya
    place_hit = False
    words = toks
    for i, w in enumerate(words[:-1]):
        if w in PLACE_PREPS:
            nxt = words[i+1]
            # heuristik: bukan stopword & bukan angka = kandidat lokasi
            if nxt not in STOPWORDS and not nxt.isdigit():
                place_hit = True; break
    return (1.0 if time_hit else 0.0, 1.0 if place_hit else 0.0)

def _pronoun_density(toks: List[str]) -> float:
    if not toks: return 0.0
    pr = sum(1 for t in toks if t in PRONOUNS)
    return pr / len(toks)

def _actionverb_density(toks: List[str]) -> float:
    if not toks: return 0.0
    av = sum(1 for t in toks if t in ACTION_VERBS)
    return av / len(toks)

def _passive_rate_id(tokens: List[str]) -> float:
    """Heuristik pasif: kata kerja berprefiks 'di-' + 'oleh' sebagai sinyal."""
    if not tokens: return 0.0
    di_verb = sum(1 for t in tokens if re.match(r"^di[a-z]", t) and t not in {"dia","di"})
    oleh = sum(1 for t in tokens if t in PASSIVE_HINTS)
    rate = (di_verb + 0.5*oleh) / max(1, len(tokens))
    # batasi agar tidak ‘menghukum’ terlalu besar
    return min(0.3, rate)

def _propn_density_with_stanza(sentences: List[str]) -> float:
    """Butuh _stanza_nlp. Propn density ~ entitas tokoh/tempat."""
    if _stanza_nlp is None or not sentences:
        return 0.0
    try:
        doc = _stanza_nlp(" ".join(sentences))
        total = 0; propn = 0
        for s in doc.sentences:
            for w in s.words:
                total += 1
                if getattr(w, "upos", "") == "PROPN":
                    propn += 1
        return propn / max(1, total)
    except Exception:
        return 0.0

PRONOUNS = {"saya", "aku", "kami", "kita", "kamu", "anda", "beliau", "dia", "mereka"}
NARRATIVE_MARKERS = ACTION_VERBS | PRONOUNS | {"kisah", "cerita", "tokoh"}

# --- Implementasi narativity dengan fitur-fitur di atas ---

# Seeds for concreteness (expand with your domain vocabulary)
CONCRETE_SEEDS = {
    # benda nyata/fisik
    "kursi", "meja", "apel", "air", "tanah", "vanili", "daun", "akar", "batang", "bunga",
    "kompor", "motor", "ban", "alat", "sensor", "kamera", "data", "peta", "rumah", "sekolah",
    "media", "tanam", "MSG", "larutan", "konsentrasi", "pertumbuhan", "biji", "buah",
    "buku", "pena", "kertas", "pintu", "jendela", "mobil", "sepeda", "jalan", "gunung",
    "laut", "sungai", "hutan", "hewan", "ikan", "ayam", "sapi", "burung", "kucing", "anjing",
    "komputer", "televisi", "radio", "lampu", "gelas", "piring", "sendok", "garpu", "tas",
    "baju", "celana", "sepatu", "topi", "jaket", "jam", "uang", "koin", "dompet", "kunci",
    "gedung", "jembatan", "taman", "pasar", "warung", "restoran", "hotel", "bandara",
    "kereta", "stasiun", "bus", "terminal", "kapal", "pelabuhan", "band", "alat musik",
    "gitar", "piano", "drum", "mikrofon", "speaker", "printer", "scanner", "proyektor",
    "kursus", "kelas", "laboratorium", "perpustakaan", "kantor", "ruang", "tempat", "kamar",
    "ranjang", "lemari", "rak", "dapur", "toilet", "kamar mandi", "atap", "lantai", "tembok",
    "pagar", "tanaman", "pot", "benih", "pupuk", "air", "ember", "selang", "sumur", "pompa",
    "sepeda motor", "mobil", "truk", "angkot", "ojek", "perahu", "kapal", "pesawat", "helikopter",
    "jalan raya", "trotoar", "lampu lalu lintas", "rambu", "halte", "parkir", "garasi",
}

ABSTRACT_SEEDS = {
    # konsep, ide, sifat, proses
    "kebebasan", "keadilan", "pertumbuhan", "kualitas", "efektivitas", "strategi", "kebijakan",
    "metode", "konsep", "teori", "proses", "pengaruh", "hubungan", "korelasi", "analisis",
    "pengetahuan", "pemahaman", "pendidikan", "kesehatan", "keamanan", "kepercayaan",
    "kesuksesan", "kegagalan", "kemajuan", "kemunduran", "perubahan", "perkembangan",
    "perencanaan", "penelitian", "penemuan", "penyelesaian", "penyebab", "akibat", "tujuan",
    "harapan", "cita-cita", "impian", "motivasi", "inspirasi", "kreativitas", "inovasi",
    "nilai", "norma", "etika", "moral", "agama", "kepercayaan", "budaya", "tradisi",
    "identitas", "karakter", "sikap", "perasaan", "emosi", "pikiran", "ide", "gagasan",
    "pendapat", "argumen", "alasan", "logika", "fakta", "opini", "teknik", "sistem",
    "struktur", "fungsi", "peran", "tanggung jawab", "hak", "kewajiban", "kebutuhan",
    "keinginan", "pilihan", "kesempatan", "tantangan", "masalah", "solusi", "resiko",
    "manfaat", "kerugian", "keuntungan", "biaya", "waktu", "proyek", "program", "agenda",
    "visi", "misi", "sasaran", "target", "indikator", "evaluasi", "monitoring", "kontrol",
}

# Clause markers (very rough heuristic for clause counting)
CLAUSE_MARKERS = {
    ",", ";", ":", 
    "yang", "bahwa", "karena", "agar", "walaupun", "meskipun", "sehingga",
    "jika", "bila", "apabila", "supaya", "untuk", "sebab", "sebabnya",
    "tetapi", "namun", "sedangkan", "sementara", "padahal", "walau", "walaupun",
    "lalu", "kemudian", "setelah", "sebelum", "hingga", "sampai", "seandainya",
    "andaikan", "asalkan", "meski", "meskipun", "oleh karena itu", "akibatnya"
}

# === Improved Sentence & Word Tokenizer (Indonesian-aware) ===

ID_ABBR = {
    "dr.", "drg.", "prof.", "ir.", "s.t.", "s.kom.", "m.kom.", "s.pd.",
    "s.si.", "m.si.", "m.sc.", "m.eng.", "m.t.", "ph.d.", "no.", "hlm.",
    "dsb.", "dll.", "dkk.", "s.d.", "u.u.d.", "ttg.", "pas.", "ayat.", "rp."
}
EN_ABBR = {"mr.", "mrs.", "ms.", "dr.", "prof.", "etc.", "e.g.", "i.e."}
SAFE_ABBR = ID_ABBR | EN_ABBR

RE_EMAIL    = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
RE_URL      = re.compile(r"\bhttps?://[^\s]+|\bwww\.[^\s]+\b", re.IGNORECASE)
RE_INITIALS = re.compile(r"(?<=\b[A-Za-z])\.(?=\s*[A-Za-z]\b)")
RE_DECIMAL  = re.compile(r"(?<=\d)\.(?=\d)")
RE_ELLIPSIS = re.compile(r"\.\.\.+")
RE_WORD     = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\-]+")

def _tok_mask(text: str) -> str:
    t = text

    def dotmask(m: re.Match) -> str:
        return m.group(0).replace(".", "<DOT>")
    t = RE_URL.sub(dotmask, t)
    t = RE_EMAIL.sub(dotmask, t)

    t = RE_ELLIPSIS.sub("<ELLIPSIS>", t)

    abbr_alt = "|".join(re.escape(a) for a in sorted(SAFE_ABBR))
    def mask_abbr(m: re.Match) -> str:
        return m.group(0)[:-1] + "<DOT>"
    t = re.sub(rf"(?i)\b(?:{abbr_alt})", mask_abbr, t)

    t = RE_INITIALS.sub("<DOT>", t)
    t = RE_DECIMAL.sub("<DOT>", t)
    return t

def _tok_unmask(text: str) -> str:
    return text.replace("<ELLIPSIS>", "…").replace("<DOT>", ".")

def sentence_split(text: str, hard_wrap_words: int = 0) -> List[str]:
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    protected = _tok_mask(text)
    parts = re.split(r"(?<=[.!?])\s+", protected)

    if hard_wrap_words and hard_wrap_words > 0:
        wrapped: List[str] = []
        for p in parts:
            ws = p.split()
            if len(ws) > hard_wrap_words:
                for i in range(0, len(ws), hard_wrap_words):
                    wrapped.append(" ".join(ws[i:i + hard_wrap_words]))
            else:
                wrapped.append(p)
        parts = wrapped

    sents = [_tok_unmask(p).strip() for p in parts if p and p.strip()]
    return sents if sents else [text]

def word_tokenize(text: str) -> List[str]:
    if not text:
        return []
    spans = []
    for m in RE_URL.finditer(text):
        spans.append((m.start(), m.end()))
    for m in RE_EMAIL.finditer(text):
        spans.append((m.start(), m.end()))
    spans.sort()
    toks: List[str] = []
    i = 0
    for s, e in spans:
        if i < s:
            toks.extend(RE_WORD.findall(text[i:s]))
        toks.append(text[s:e])
        i = e
    if i < len(text):
        toks.extend(RE_WORD.findall(text[i:]))

    return [t.lower() for t in toks if t]

def syllable_count_id(word: str) -> int:
    if not word:
        return 1
    groups = re.findall(r"[aiueo]+", word.lower())
    return max(1, len(groups))
# ----------------------------
# Tokenization & syllables
# ----------------------------
def sentence_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    protected = text.replace("dr.", "dr<dot>").replace("s.d.", "s<dot>d<dot>")
    protected = protected.replace("dll.", "dll<dot>").replace("dsb.", "dsb<dot>")
    
    # Pisah berdasarkan tanda baca + spasi atau akhir teks
    parts = re.split(r"(?<=[.!?;:])\s+", protected)
    
    # Tambahan: pisah jika terlalu panjang (>30 kata) dan tidak ada tanda baca
    extended = []
    for p in parts:
        words = p.split()
        if len(words) > 30:
            chunks = [" ".join(words[i:i+30]) for i in range(0, len(words), 30)]
            extended.extend(chunks)
        else:
            extended.append(p)
    
    sents = [p.replace("<dot>", ".").strip() for p in extended if p.strip()]
    return sents if sents else ([text] if text else [])


def word_tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_\-]+", text.lower())

def syllable_count_id(word: str) -> int:
    groups = re.findall(r"[aiueo]+", word.lower())
    return max(1, len(groups))

# ----------------------------
# Normalization helpers (0–100)
# ----------------------------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def to_0_100(x: float) -> float:
    return round(clamp01(x) * 100.0, 1)

def inv_scale(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0
    x = max(lo, min(hi, x))
    return 1.0 - (x - lo) / (hi - lo)

# ----------------------------
# Metric computations
# ----------------------------

def narrativity(sentences):
    if not sentences:
        return 0.0
    text = " ".join(sentences)
    toks = word_tokenize(text)
    if not toks:
        return 0.0
    pron = _pronoun_density(toks)
    action = _actionverb_density(toks)
    time_hit, place_hit = _has_time_place(text, toks)
    char_propn = _propn_density_with_stanza(sentences)
    passive_penalty = _passive_rate_id(toks)
    base = (
        0.20 * pron +
        0.30 * action +
        0.10 * time_hit +
        0.10 * place_hit +
        0.10 * char_propn
    )
    # add small baseline and subtract limited passive penalty
    score = max(0.0, min(1.0, base - 0.5 * passive_penalty + 0.20))
    return to_0_100(score)
TIME_WORDS = {
    "kemarin","tadi","besok","nanti","sekarang","pada","pukul","jam",
    "minggu","bulan","tahun","januari","februari","maret","april","mei","juni",
    "juli","agustus","september","oktober","november","desember","hari","senin",
    "selasa","rabu","kamis","jumat","sabtu","minggu"
}
PLACE_PREPS = {"di","ke","dari"}
PASSIVE_HINTS = {"oleh"}  # + prefix 'di-' di kata kerja

def _has_time_place(text: str, toks: List[str]) -> Tuple[float, float]:
    tset = set(toks)
    time_hit = any(w in tset for w in TIME_WORDS) or bool(re.search(r"\b(20\d{2}|19\d{2})\b", text))
    # tempat: preposisi tempat + kata setelahnya
    place_hit = False
    words = toks
    for i, w in enumerate(words[:-1]):
        if w in PLACE_PREPS:
            nxt = words[i+1]
            # heuristik: bukan stopword & bukan angka = kandidat lokasi
            if nxt not in STOPWORDS and not nxt.isdigit():
                place_hit = True; break
    return (1.0 if time_hit else 0.0, 1.0 if place_hit else 0.0)

def _pronoun_density(toks: List[str]) -> float:
    if not toks: return 0.0
    pr = sum(1 for t in toks if t in PRONOUNS)
    return pr / len(toks)

def _actionverb_density(toks: List[str]) -> float:
    if not toks: return 0.0
    av = sum(1 for t in toks if t in ACTION_VERBS)
    return av / len(toks)

def _passive_rate_id(tokens: List[str]) -> float:
    """Heuristik pasif: kata kerja berprefiks 'di-' + 'oleh' sebagai sinyal."""
    if not tokens: return 0.0
    di_verb = sum(1 for t in tokens if re.match(r"^di[a-z]", t) and t not in {"dia","di"})
    oleh = sum(1 for t in tokens if t in PASSIVE_HINTS)
    rate = (di_verb + 0.5*oleh) / max(1, len(tokens))
    # batasi agar tidak ‘menghukum’ terlalu besar
    return min(0.3, rate)

def _propn_density_with_stanza(sentences: List[str]) -> float:
    """Butuh _stanza_nlp. Propn density ~ entitas tokoh/tempat."""
    if _stanza_nlp is None or not sentences:
        return 0.0
    try:
        doc = _stanza_nlp(" ".join(sentences))
        total = 0; propn = 0
        for s in doc.sentences:
            for w in s.words:
                total += 1
                if getattr(w, "upos", "") == "PROPN":
                    propn += 1
        return propn / max(1, total)
    except Exception:
        return 0.0

PRONOUNS = {"saya", "aku", "kami", "kita", "kamu", "anda", "beliau", "dia", "mereka"}
NARRATIVE_MARKERS = ACTION_VERBS | PRONOUNS | {"kisah", "cerita", "tokoh"}

# --- Implementasi narativity dengan fitur-fitur di atas ---

def syntactic_simplicity(sentences: List[str]) -> float:
    """If Stanza is available, use dependency-based complexity; else fallback to heuristic."""
    if not sentences:
        return 100.0

    if _stanza_nlp is not None:
        try:
            doc = _stanza_nlp(" ".join(sentences))
            sent_scores = []
            for s in doc.sentences:
                children = {i+1: [] for i in range(len(s.words))}
                for i, w in enumerate(s.words, start=1):
                    if w.head in children:
                        children[w.head].append(i)
                def depths(start: int, d: int, acc: List[int]):
                    acc.append(d)
                    for ch in children.get(start, []):
                        depths(ch, d+1, acc)
                all_depths: List[int] = []
                depths(0, 0, all_depths)  # roots have head=0
                avg_depth = sum(all_depths) / max(1, len(all_depths))
                subcls = sum(1 for w in s.words if w.deprel in {"advcl","ccomp","xcomp","acl","mark","conj"})
                sent_len = len(s.words)
                depth_simple = inv_scale(avg_depth, 1.0, 6.0)
                sub_simple = inv_scale(subcls, 0.0, 4.0)
                len_simple = inv_scale(sent_len, 8.0, 35.0)
                sent_scores.append(0.4*depth_simple + 0.35*sub_simple + 0.25*len_simple)
            return to_0_100(sum(sent_scores)/max(1,len(sent_scores)))
        except Exception:
            pass  # fall back

    # Fallback heuristic
    tok_sents = [word_tokenize(s) for s in sentences]
    lens = [len(ts) for ts in tok_sents if ts]
    if not lens:
        return 100.0
    avg_len = sum(lens) / len(lens)
    clause_counts = []
    for s in sentences:
        tokens = set(word_tokenize(s))
        count = 0
        for m in CLAUSE_MARKERS:
            if m in {",", ";", ":"}:
                count += s.count(m)
            else:
                count += (m in tokens)
        clause_counts.append(count)
    avg_clause = sum(clause_counts) / max(1, len(clause_counts))
    len_simple = inv_scale(avg_len, 8.0, 30.0)
    clause_simple = inv_scale(avg_clause, 0.0, 4.0)
    simp = 0.7 * len_simple + 0.3 * clause_simple
    return to_0_100(simp)

def word_concreteness(tokens: List[str]) -> float:
    if not tokens:
        return 50.0
    content = [t for t in tokens if t not in STOPWORDS]
    if not content:
        return 50.0
    concrete = sum(1 for t in content if t in CONCRETE_SEEDS)
    abstract = sum(1 for t in content if t in ABSTRACT_SEEDS)
    base = (concrete - abstract)
    score = 0.5 + 0.5 * (base / max(1.0, len(content) ** 0.7))
    return to_0_100(score)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))

def referential_cohesion(sentences: List[str]) -> float:
    if not sentences:
        return 0.0

    if _sbert_model is not None and len(sentences) > 1:
        try:
            embs = _sbert_model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
            sims = [float(embs[i].dot(embs[i+1])) for i in range(len(embs)-1)]  # cosine with normalized vectors
            sims01 = [(s+1)/2 for s in sims]
            return to_0_100(sum(sims01)/len(sims01))
        except Exception:
            pass

    content_sents = []
    for s in sentences:
        toks = [t for t in word_tokenize(s) if t not in STOPWORDS and not t.isdigit()]
        content_sents.append(set(toks))
    if len(content_sents) == 1:
        return 20.0
    overlaps = [jaccard(content_sents[i], content_sents[i+1]) for i in range(len(content_sents)-1)]
    avg_overlap = sum(overlaps) / len(overlaps)
    return to_0_100(avg_overlap)


def deep_cohesion(text: str) -> float:
    t = text.lower()
    counts = 0
    for c in CONNECTIVES:  # <- tidak pakai .values()
        matches = re.findall(rf"(?<!\w){re.escape(c)}(?!\w)", t)
        counts += len(matches)
    num_sents = max(1, len(sentence_split(text)))
    return counts / num_sents




def inv_scale(x: float, low: float, high: float) -> float:
    """Skala invers antara low dan high, dibatasi 0–1."""
    return max(0.0, min(1.0, (high - x) / (high - low)))

def flesch_like_id(sentences: List[str]) -> Tuple[float, float, str]:
    if not sentences:
        return 100.0, 100.0, "Mudah"
    
    text = " ".join(sentences)
    toks = word_tokenize(text)
    if not toks:
        return 100.0, 100.0, "Mudah"
    
    words = len(toks)
    sents = len(sentences)
    syll = sum(syllable_count_id(w) for w in toks)

    wps = words / max(1, sents)  # words per sentence
    spw = syll / max(1, words)  # syllables per word

    # Rentang disesuaikan untuk toleransi teks ilmiah
    ease = (
        inv_scale(wps, 10.0, 40.0) * 0.6 +
        inv_scale(spw, 1.3, 2.5) * 0.4
    )

    raw = ease * 100.0
    norm = to_0_100(ease)

    if norm >= 80:
        cat = "Sangat Mudah"
    elif norm >= 60:
        cat = "Mudah"
    elif norm >= 40:
        cat = "Sedang"
    elif norm >= 20:
        cat = "Cukup Sulit"
    else:
        cat = "Sangat Sulit"

    return raw, norm, cat


@dataclass
class TeraScores:
    Narrativity: float
    Syntactic_Simplicity: float
    Word_Concreteness: float
    Referential_Cohesion: float
    Deep_Cohesion: float
    Flesch_Score_Raw: float
    Flesch_Score_Normalized: float
    Flesch_Category: str

def score_text(text: str) -> Dict[str, object]:
    text = (text or "").strip()
    sents = sentence_split(text)
    tokens = word_tokenize(text)

    narr = narrativity(sents)
    synt = syntactic_simplicity(sents)
    conc = word_concreteness(tokens)
    refc = referential_cohesion(sents)
    deep = deep_cohesion(text)
    flesch_raw, flesch_norm, flesch_cat = flesch_like_id(sents)

    scores = TeraScores(
        Narrativity=narr,
        Syntactic_Simplicity=synt,
        Word_Concreteness=conc,
        Referential_Cohesion=refc,
        Deep_Cohesion=deep,
        Flesch_Score_Raw=flesch_raw,
        Flesch_Score_Normalized=flesch_norm,
        Flesch_Category=flesch_cat,
    )
    return asdict(scores)

# ----------------------------
# CSV helpers
# ----------------------------
def score_csv(in_path: str, text_col: str, out_path: str) -> None:
    rows = []
    with open(in_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if text_col not in reader.fieldnames:
            raise ValueError(f"Column '{text_col}' not found. Available: {reader.fieldnames}")
        for i, row in enumerate(reader, start=1):
            text = row.get(text_col, "")
            scores = score_text(text)
            out_row = {"ID": i, "Teks": text}
            out_row.update(scores)
            rows.append(out_row)
    fieldnames = ["ID", "Teks", "Narrativity", "Syntactic_Simplicity", "Word_Concreteness",
                  "Referential_Cohesion", "Deep_Cohesion", "Flesch_Score_Raw",
                  "Flesch_Score_Normalized", "Flesch_Category"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Mini TERA-style Readability for Indonesian")
    parser.add_argument("text", nargs="?", help="Single text to score")
    parser.add_argument("--csv", dest="csv_in", help="Input CSV path")
    parser.add_argument("--text-col", dest="text_col", default="Teks", help="Column name containing text")
    parser.add_argument("--out", dest="csv_out", default="scores.csv", help="Output CSV path")
    parser.add_argument("--stanza-dir", dest="stanza_dir", default=STANZA_DIR, help="Path to local stanza_resources")
    parser.add_argument("--sbert-dir", dest="sbert_dir", default=SBERT_DIR, help="Path to local SBERT cache/model")
    args = parser.parse_args()

    # If custom dirs are passed at runtime, try reload with those dirs
    global _stanza_nlp, _sbert_model
    if args.stanza_dir and os.path.abspath(args.stanza_dir) != os.path.abspath(STANZA_DIR):
        try:
            os.environ["STANZA_RESOURCES_DIR"] = args.stanza_dir
            import stanza as _st
            _stanza_nlp = _st.Pipeline('id', processors='tokenize,pos,lemma,depparse',
                                       tokenize_no_ssplit=False, verbose=False, dir=args.stanza_dir)
        except Exception:
            pass

    if args.sbert_dir and os.path.abspath(args.sbert_dir) != os.path.abspath(SBERT_DIR):
        try:
            from sentence_transformers import SentenceTransformer as _SB
            local_path = os.path.join(args.sbert_dir, 'paraphrase-multilingual-MiniLM-L12-v2')
            if os.path.isdir(local_path):
                _sbert_model = _SB(local_path)
            else:
                _sbert_model = _SB('paraphrase-multilingual-MiniLM-L12-v2', cache_folder=args.sbert_dir)
        except Exception:
            _sbert_model = None


    if args.csv_in:
        score_csv(args.csv_in, args.text_col, args.csv_out)
        print(f"Saved scores to {args.csv_out}")

    sample = args.text or "Penelitian bertujuan untuk mengetahui pengaruh komposisi media tanam dan pemberian konsentrasi MSG terhadap pertumbuhan vanili (Vanilla planifolia) stek satu ruas."
    print(score_text(sample))

if __name__ == "__main__":
    main()
