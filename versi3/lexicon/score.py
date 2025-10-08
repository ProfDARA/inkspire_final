# file: versi3/lexicon/score_text_metrics.py
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

# ===================== Kamus =====================
STOPWORDS = set(
    "yang di ke dari kepada untuk dengan pada oleh demi agar supaya sehingga karena namun tetapi sedangkan atau dan serta pun bila jika ketika maka lalu kemudian sehingga yaitu yakni adalah ialah seperti sebagai dalam terhadap antara tanpa bukan bukanlah pun ini itu itu pun tersebut para para nya nya pun saya aku kami kita kamu Anda anda beliau dia mereka sini sana situ itu pun hal bhw bahwa hingga hingga pun tiap setiap tiapnya seluruh seluruhnya beberapa berbagai pun akan telah telahlah telahpun sudah sudahpun telah sudah pernah sedang sedanglah sedangpun telahlah belum belumlah belumpun masih masihlah masihpun akan punlah dapat bisa mungkin tentu harus wajib tak tidak bukan bukanlah jangan ya pun punlah punnya walau walaupun meski meskipun sebaliknya sekalipun selainnya selain selainpun hingga ketika saat waktu sewaktu seraya sambil sambilpun sementara sementara itu kemudian lantas lalu selanjutnya berikutnya selaku selagi selama sehabis seusai setelah sebelum sebelumya sebelum pun sesudah sesudahnya maka yaitu yakni sebab karenanya akibatnya oleh karena itu contohnya misalnya misal".split()
)

CONNECTIVES = {
    # causal
    "karena","sebab","akibatnya","sehingga","maka",
    "oleh","oleh karena itu","oleh sebab itu","disebabkan oleh",
    "yang menyebabkan","yang mengakibatkan","dengan demikian",
    "sebagai akibat","konsekuensinya","sebagai konsekuensi","dampaknya","implikasinya","berkat",
    # contrast
    "namun","tetapi","meski","meskipun","walau",
    "sebaliknya","padahal","berbeda dengan","di sisi lain",
    "kendati demikian","walaupun demikian","akan tetapi",
    "namun demikian","meskipun demikian","sekalipun demikian",
    "bertolak belakang dengan",
    # temporal
    "kemudian","lalu","selanjutnya","berikutnya",
    "sebelumnya","setelah","sebelum","setelah itu",
    "pada tahap berikutnya","pada akhirnya","selama",
    "ketika","saat","seiring dengan","hingga",
    "sementara itu","pada saat yang sama","di kemudian hari",
    "sepanjang","sejak itu","di awal","di akhir",
    # exemplify
    "misal","misalnya","contohnya","sebagai contoh",
    "yakni","yaitu","dengan kata lain","antara lain",
    "seperti","termasuk","sebagai ilustrasi","contoh kasus","khususnya",
    # additive
    "selain itu","di samping itu","lebih lanjut","lebih jauh","lagipula",
    # conclusion
    "sebagai kesimpulan","kesimpulannya","dapat disimpulkan bahwa","ringkasnya","singkatnya","dari paparan tersebut",
    # conditional
    "jika","apabila","bila","seandainya","asalkan","jikalau","andaikan","kecuali jika",
    # purpose
    "agar","supaya","guna","dalam rangka",
    # comparison
    "dibandingkan dengan","dibanding dengan","seperti halnya","serupa dengan","berkebalikan dengan",
    # definition
    "didefinisikan sebagai","yang dimaksud dengan","dalam konteks ini","dalam hal ini",
    # attribution
    "berdasarkan","menurut","mengacu pada","merujuk pada",
    # limitation
    "terbatas pada","hanya saja","sebatas"
}

ACTION_VERBS = {
    "pergi","datang","pulang","berjalan","berlari","melompat","masuk","keluar","naik","turun",
    "mendekat","menjauh","kembali","berputar","berhenti","bergerak",
    "berkata","bercerita","menceritakan","mengisahkan","menjelaskan","menjawab","bertanya",
    "mengungkapkan","mengucapkan","menyebutkan","memberitahu","menyampaikan","menegaskan",
    "menambahkan","mengulangi","mendengar","menyimak","menanggapi","membantah",
    "melihat","menatap","memandang","memperhatikan","mengamati","menemukan","menyadari",
    "mengingat","membayangkan","memikirkan","menganalisis","menilai","mengukur","menghitung",
    "mengidentifikasi","mencatat","menafsirkan","mengecek","memeriksa","meneliti","mengumpulkan",
    "menyimpulkan","membandingkan","memutuskan","memilih","merencanakan","mengembangkan",
    "mengambil","mengerjakan","menaruh","meletakkan","membawa","memindahkan","membuka","menutup","memasukkan",
    "mengeluarkan","menyusun","menyiapkan","membuat","membangun","mencetak","memotong",
    "mengaduk","mencampur","memasak","menggoreng","memanggang","mencuci","membersihkan",
    "mengeringkan","mengganti","memperbaiki","memasang","melepas","mengikat","menggosok",
    "menekan","menarik","mendorong","menggenggam","menyalakan","mematikan","menghidupkan",
    "menulis","menyalin","mengedit","menghapus","mengunggah","mengunduh","menyimpan",
    "mengirim","mengirimkan","memprogram","mengkode","menjalankan","mengoperasikan",
    "terjadi","muncul","timbul","membesar","mengecil","berubah","menghilang","membeku",
    "mencair","mendadak","membentuk","menghasilkan","mengakibatkan","menyebabkan",
    "membantu","menolong","menyelamatkan","menjaga","melindungi","mengawal","menemani",
    "menghadapi","menyerang","menolak","menerima","menegur","memarahi","menghibur",
    "menghadiri","mengikuti","berkumpul","bertemu","berdiskusi","berkolaborasi",
    "tersenyum","tertawa","menangis","meratap","mengeluh","menghela","menarik napas",
    "terkejut","kaget","marah","kesal","bangga","gugup","cemas","tenang","lega","sedih",
    "mencoba","berhasil","gagal","mencapai","meraih","mendapatkan","kehilangan","menemukan",
    "mengumpulkan","menyelesaikan","melanjutkan","menghentikan","mengakhiri","memulai",
    "mencari","menemukan","menelusuri","mengejar","melacak","mengikuti","mengawasi",
    "menunggu","menjadwalkan","menunda","mempercepat","melanjutkan","mengulang","mengawali",
    "makan","memakan","minum","meminum","tidur","bangun","bernapas","beristirahat",
    "berjuang","bertahan","menyerah","menyatu","memisahkan","menghubungkan","menyatukan",
    "mengendalikan","menguasai","mengelola","mengurangi","menambah","meningkatkan",
    "menurunkan","mempercepat","memperlambat","menyaring","mengurutkan","menyalurkan"
}

TIME_WORDS = {
    "kemarin","tadi","besok","nanti","sekarang","pada","pukul","jam",
    "minggu","bulan","tahun","januari","februari","maret","april","mei","juni",
    "juli","agustus","september","oktober","november","desember","hari","senin",
    "selasa","rabu","kamis","jumat","sabtu","minggu"
}
PLACE_PREPS = {"di","ke","dari"}
PASSIVE_HINTS = {"oleh"}  # + prefiks 'di-' pada verba

# ===================== Util =====================
SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+')
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-žΑ-ωА-яİıĞğŞşÇçŊŋ']+")
VOWELS = set("aiueoAIUEO")

def clean(s): return str(s).strip()
def tokens(text): return [w.lower() for w in WORD_RE.findall(str(text))]

def sentence_similarity(embedder, a:str, b:str) -> float:
    if not a or not b: return 0.0
    E = embedder.encode([a,b], convert_to_numpy=True, normalize_embeddings=True)
    return float(np.clip((E[0]*E[1]).sum(), -1.0, 1.0))

def content_overlap(a:str, b:str)->float:
    ta = {w for w in tokens(a) if w not in STOPWORDS}
    tb = {w for w in tokens(b) if w not in STOPWORDS}
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

def ratio_in_vocab(tok:list, vocab:set)->float:
    n = len(tok) or 1
    return sum(t in vocab for t in tok) / n

def syllable_guess(word:str)->int:
    w = str(word)
    c, pv = 0, False
    for ch in w:
        v = ch in VOWELS
        if v and not pv: c += 1
        pv = v
    return max(1, c)

def split_sentences(text: str):
    s = str(text).strip()
    if not s:
        return []
    sents = [t.strip() for t in SENT_SPLIT_RE.split(s) if t.strip()]
    return sents if sents else [s]

def flesch_normalized(text: str) -> float:
    sents = split_sentences(text)
    n_sent = max(1, len(sents))
    tok = tokens(text)
    n_words = max(1, len(tok))
    n_syll = sum(syllable_guess(w) for w in tok)
    fre = 206.835 - 1.015 * (n_words / n_sent) - 84.6 * (n_syll / n_words)
    return float(np.clip(fre, 0, 100))

# ===================== Concreteness 0–5 → 0–100 =====================
def concretize_tokens_0_100(tok:list, conc_map:dict):
    vals = [float(conc_map[t]) for t in tok if t in conc_map]
    if not vals:
        return (np.nan, np.nan, np.nan)
    mn = min(vals); mx = max(vals)
    if 0.0 <= mn <= 5.0 and mx <= 5.0 and mn == 0.0:
        scaled = [v/5.0*100.0 for v in vals]              # 0..5
    elif 1.0 <= mn <= 5.0 and mx <= 5.0:
        scaled = [((v-1.0)/4.0)*100.0 for v in vals]      # 1..5
    else:
        a,b = (mn, mx) if mx>mn else (0.0, 1.0)
        scaled = [((v-a)/(b-a))*100.0 for v in vals]      # fallback
    mean_100 = float(np.mean(scaled))
    concrete_ratio = mean_100 / 100.0
    abstract_ratio = 1.0 - concrete_ratio
    return (mean_100, concrete_ratio, abstract_ratio)

# ===================== Skor metrik =====================
def syntactic_simplicity(text:str)->float:
    tok = tokens(text)
    n_tok = len(tok)
    if n_tok == 0: return 100.0
    avg_wlen = np.mean([len(t) for t in tok])
    punct_density = sum(ch in ",;:" for ch in text) / max(1, len(text))
    clause_ratio = ratio_in_vocab(tok, {"yang","bahwa","ketika","jika","apabila","sehingga","karena","walaupun","meskipun"})
    len_term = 1.0 - np.tanh((n_tok-12)/30)
    wlen_term = 1.0 - np.tanh((avg_wlen-5.0)/4.0)
    punct_term = 1.0 - 4.0*punct_density
    clause_term = 1.0 - 2.0*clause_ratio
    score = 100 * np.clip(0.35*len_term + 0.25*wlen_term + 0.20*punct_term + 0.20*clause_term, 0, 1)
    return float(np.clip(score, 0, 100))

def narrativity(text:str, concrete_ratio:float)->float:
    tok = tokens(text)
    pron = ratio_in_vocab(tok, {"aku","saya","kami","kita","kamu","dia","mereka","dirinya"})
    act  = ratio_in_vocab(tok, ACTION_VERBS)
    time = ratio_in_vocab(tok, TIME_WORDS)
    place = ratio_in_vocab(tok, PLACE_PREPS)
    passive = ratio_in_vocab(tok, PASSIVE_HINTS)
    conc = (concrete_ratio if not np.isnan(concrete_ratio) else 0.5)
    score = 100 * np.clip(
        0.25*conc + 0.20*pron + 0.20*act + 0.10*time + 0.10*place - 0.05*passive, 0, 1
    )
    return float(score)

def ref_cohesion_local(prev_text:str, cur_text:str, next_text:str, embedder)->float:
    sims, overs = [], []
    if prev_text:
        sims.append(sentence_similarity(embedder, prev_text, cur_text))
        overs.append(content_overlap(prev_text, cur_text))
    if next_text:
        sims.append(sentence_similarity(embedder, cur_text, next_text))
        overs.append(content_overlap(cur_text, next_text))
    if not sims: return 0.0
    return float(100 * np.clip(0.55*np.mean(overs) + 0.45*np.mean(sims), 0, 1))

def deep_cohesion_local(prev_text:str, cur_text:str, next_text:str, embedder)->float:
    tok = tokens(cur_text)
    text_l = cur_text.lower()
    conn_hit = any(c in text_l for c in CONNECTIVES)
    conn_ratio = max(ratio_in_vocab(tok, CONNECTIVES), 0.05 if conn_hit else 0.0)
    sims = []
    if prev_text: sims.append(sentence_similarity(embedder, prev_text, cur_text))
    if next_text: sims.append(sentence_similarity(embedder, cur_text, next_text))
    sim_term = np.mean(sims) if sims else 0.0
    return float(100 * np.clip(0.6*conn_ratio + 0.4*sim_term, 0, 1))

# ===================== Main =====================
def main():
    ap = argparse.ArgumentParser(description="Skoring Narativitas, Referential/Deep Cohesion per kalimat + injeksi Word Concreteness (0–5 => 0–100).")
    ap.add_argument("--kata-csv", required=True, help="CSV peta kata → concreteness (Kata,concreteness_pred,...)")
    ap.add_argument("--raw-csv", required=True, help="CSV kalimat mentah (Kalimat,Judul,Penulis,Jurusan)")
    ap.add_argument("--out", default="output_naratif.csv", help="Path output CSV")
    ap.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2", help="SentenceTransformer untuk kemiripan")
    args = ap.parse_args()

    # 1) Peta concreteness
    df_k = pd.read_csv(args.kata_csv, encoding="utf-8-sig", skipinitialspace=True)
    df_k.columns = [c.strip() for c in df_k.columns]
    if "Kata" not in df_k.columns or "concreteness_pred" not in df_k.columns:
        raise ValueError("File kata harus punya kolom 'Kata' dan 'concreteness_pred'.")
    conc_map = {str(k).strip().lower(): float(v) for k, v in zip(df_k["Kata"], df_k["concreteness_pred"]) if pd.notna(k) and pd.notna(v)}

    # 2) Data kalimat
    df = pd.read_csv(args.raw_csv, encoding="utf-8-sig", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    for col in ["Kalimat","Judul","Penulis","Jurusan"]:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di raw-csv.")
    df["_row_idx"] = np.arange(len(df))

    # 3) Embedder untuk kemiripan
    embedder = SentenceTransformer(args.model)

    # 4) Skoring per judul
    out_rows = []
    df_sorted = df.sort_values(["Judul","_row_idx"], kind="stable")
    groups = df_sorted.groupby("Judul", sort=False)

    for judul, g in groups:
        inds = list(g.index)
        for pos, i in enumerate(inds):
            row = df_sorted.loc[i]
            kal = clean(row["Kalimat"])
            tok = tokens(kal)

            mean100, conc_ratio, abs_ratio = concretize_tokens_0_100(tok, conc_map)
            prev_kal = clean(df_sorted.loc[inds[pos-1], "Kalimat"]) if pos-1 >= 0 else ""
            next_kal = clean(df_sorted.loc[inds[pos+1], "Kalimat"]) if pos+1 < len(inds) else ""

            narr = narrativity(kal, conc_ratio)
            refc = ref_cohesion_local(prev_kal, kal, next_kal, embedder)
            deep = deep_cohesion_local(prev_kal, kal, next_kal, embedder)
            syns = syntactic_simplicity(kal)
            flesch_norm = flesch_normalized(kal)

            out_rows.append({
                "_row_idx": row["_row_idx"],
                "Kalimat": row["Kalimat"],
                "Judul": row["Judul"],
                "Penulis": row["Penulis"],
                "Jurusan": row["Jurusan"],
                "Word_Concreteness_Mean_100": np.round(mean100, 2) if not np.isnan(mean100) else np.nan,
                "Concrete_Ratio": np.round(conc_ratio, 4) if not np.isnan(conc_ratio) else np.nan,
                "Abstract_Ratio": np.round(abs_ratio, 4) if not np.isnan(abs_ratio) else np.nan,
                "Narrativity": np.round(narr, 2),
                "Referential_Cohesion": np.round(refc, 2),
                "Deep_Cohesion": np.round(deep, 2),
                "Syntactic_Simplicity": np.round(syns, 2),
                "Flesch_Normalized": np.round(flesch_norm, 2),
            })

    out_df = pd.DataFrame(out_rows).sort_values("_row_idx", kind="stable").drop(columns=["_row_idx"])
    base_cols = ["Kalimat","Judul","Penulis","Jurusan"]
    score_cols = [
        "Word_Concreteness_Mean_100","Concrete_Ratio","Abstract_Ratio",
        "Narrativity","Referential_Cohesion","Deep_Cohesion",
        "Syntactic_Simplicity","Flesch_Normalized"
    ]
    out_df = out_df[base_cols + score_cols]

    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Saved → {out_path} (rows={len(out_df)})")

if __name__ == "__main__":
    main()
