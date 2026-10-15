
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="DapurFinder | Temukan Resep",
    page_icon="🍲",
    layout="wide"
)

# Perbaikan kontras dropdown dan tampilan
st.markdown("""
<style>
/* Teks dropdown selalu gelap */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid #cbdac8 !important;
    border-radius: 12px !important;
}

div[data-baseweb="select"] span,
div[data-baseweb="select"] input,
div[data-baseweb="select"] div {
    color: #263d2d !important;
    -webkit-text-fill-color: #263d2d !important;
}

/* Panah dropdown */
div[data-baseweb="select"] svg {
    fill: #315d3e !important;
    color: #315d3e !important;
}

/* Menu pilihan */
div[data-baseweb="popover"],
div[data-baseweb="menu"],
div[role="listbox"],
div[role="option"] {
    background-color: #ffffff !important;
    color: #263d2d !important;
}

div[role="option"]:hover {
    background-color: #edf4e9 !important;
}

/* Label kategori dan jumlah hasil */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p {
    color: #31513a !important;
}

/* Tombol pencarian */
.stFormSubmitButton button {
    background-color: #2f7548 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
}

.stFormSubmitButton button p,
.stFormSubmitButton button span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Banner dan gambar resep */
[data-testid="stImage"] img {
    border-radius: 18px !important;
}

/* Panel bahan dan langkah memasak */
[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid #dce6d8 !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] summary {
    background-color: #edf4e9 !important;
    color: #254d34 !important;
}

[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    color: #254d34 !important;
}

[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] li {
    color: #263d2d !important;
}
</style>
""", unsafe_allow_html=True)


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "dataset" / "recipes_clean.csv"
ASSET_DIR = ROOT_DIR / "assets"
RECIPE_IMAGE_DIR = ASSET_DIR / "recipes"

# Foto berdasarkan kategori resep
FOTO_KATEGORI = {
    "ayam": RECIPE_IMAGE_DIR / "ayam.jpg",
    "tempe": RECIPE_IMAGE_DIR / "tempe.jpg",
    "telur": RECIPE_IMAGE_DIR / "telur.jpg",
    "ikan": RECIPE_IMAGE_DIR / "ikan.jpg",
    "sapi": RECIPE_IMAGE_DIR / "sapi.jpg",
    "tahu": RECIPE_IMAGE_DIR / "tempe.jpg",
    "udang": RECIPE_IMAGE_DIR / "ikan.jpg",
    "kambing": RECIPE_IMAGE_DIR / "sapi.jpg",
}

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f6f5eb 0%, #fffdf8 55%, #f4f6ee 100%);
    color: #26382c;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #234d35 !important;
    letter-spacing: -0.4px;
}

p, label, [data-testid="stMarkdownContainer"] {
    color: #35483a;
}

[data-testid="stCaptionContainer"] {
    color: #657466 !important;
}

/* Input pencarian */
.stTextInput input {
    background: #ffffff !important;
    color: #24372a !important;
    -webkit-text-fill-color: #24372a !important;
    border: 1px solid #d5dfd1 !important;
    border-radius: 12px !important;
    min-height: 46px;
}

.stTextInput input::placeholder {
    color: #7c897e !important;
    -webkit-text-fill-color: #7c897e !important;
}

/* Dropdown */
div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border-color: #d5dfd1 !important;
    border-radius: 12px !important;
}

div[data-baseweb="select"] span,
div[data-baseweb="select"] input {
    color: #26382c !important;
    -webkit-text-fill-color: #26382c !important;
}

div[data-baseweb="popover"],
div[data-baseweb="menu"],
div[role="listbox"] {
    background: #ffffff !important;
}

div[role="option"] {
    color: #26382c !important;
}

/* Form */
[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid #e1e6d8;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 8px 25px rgba(38, 61, 39, 0.06);
}

/* Tombol */
.stFormSubmitButton button,
.stLinkButton a {
    background: #2f7048 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 11px !important;
    font-weight: 650 !important;
}

.stFormSubmitButton button:hover,
.stLinkButton a:hover {
    background: #225738 !important;
    color: #ffffff !important;
}

/* Kartu resep */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff;
    border: 1px solid #e3e8dc;
    border-radius: 16px;
    box-shadow: 0 5px 18px rgba(37, 65, 41, 0.045);
}

/* Expander bahan dan langkah */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #dfe6da !important;
    border-radius: 12px !important;
    overflow: hidden;
}

[data-testid="stExpander"] summary {
    background: #f0f5ec !important;
    color: #244d35 !important;
}

[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p {
    color: #244d35 !important;
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: #ffffff !important;
    color: #26382c !important;
}

[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] li {
    color: #26382c !important;
}

/* Pemberitahuan */
[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: #e1e7dc !important;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem 1rem 2rem;
    }
    h1 {
        font-size: 2rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def muat_data(path):
    data = pd.read_csv(path).fillna("")
    data["search_text"] = (
        data["Title"].astype(str) + " "
        + data["Ingredients"].astype(str)
    )
    return data


@st.cache_resource
def bangun_mesin_pencarian(teks):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(teks)
    return vectorizer, matrix


def tampilkan_daftar(teks):
    teks = str(teks).strip()
    if not teks:
        return ["Belum ada informasi."]
    if "--" in teks:
        bagian = [x.strip(" -\n\t") for x in teks.split("--")]
    else:
        bagian = [x.strip(" -\n\t") for x in teks.splitlines()]
    return [x for x in bagian if x] or ["Belum ada informasi."]


def cari_foto(kategori, judul=""):
    nama_kategori = str(kategori).lower()
    nama_judul = str(judul).lower()

    # Utamakan kategori, lalu cek kata pada judul
    for nama, path in FOTO_KATEGORI.items():
        if nama in nama_kategori and path.exists():
            return path

    for nama, path in FOTO_KATEGORI.items():
        if nama in nama_judul and path.exists():
            return path

    # Foto cadangan untuk resep yang kategorinya belum dipetakan
    foto_cadangan = RECIPE_IMAGE_DIR / "tempe.jpg"
    if foto_cadangan.exists():
        return foto_cadangan
    return None


if not DATA_PATH.exists():
    st.error(
        "Dataset tidak ditemukan. Pastikan file "
        "dataset/recipes_clean.csv tersedia."
    )
    st.stop()

df = muat_data(DATA_PATH)
vectorizer, matrix = bangun_mesin_pencarian(
    df["search_text"].tolist()
)

# Banner utama
kolom_teks, kolom_foto = st.columns([1.25, 1], gap="large")

with kolom_teks:
    st.markdown(
        "<div style='padding-top:20px; color:#6c805f; "
        "font-weight:700; letter-spacing:2px;'>"
        "TEMUKAN INSPIRASI DI DAPUR</div>",
        unsafe_allow_html=True
    )
    st.title("🍲 DapurFinder")
    st.markdown(
        "<div style='font-size:19px; font-weight:600; color:#315a3e;'>"
        "Masak enak, mulai dari bahan yang kamu punya.</div>",
        unsafe_allow_html=True
    )
    st.write(
        "Cari resep berdasarkan nama masakan atau bahan makanan. "
        "DapurFinder menggunakan TF-IDF dan cosine similarity "
        "untuk menemukan resep yang paling relevan."
    )
    st.caption(f"📚 {len(df):,} resep tersedia untuk kamu jelajahi.")

with kolom_foto:
    foto_hero = ASSET_DIR / "hero-masakan.jpg"
    if foto_hero.exists():
        st.image(foto_hero, use_container_width=True)
    else:
        st.info("Tambahkan assets/hero-masakan.jpg untuk menampilkan banner.")

st.markdown("### 🔎 Mau masak apa hari ini?")

with st.form("form_pencarian"):
    kolom_cari, kolom_kategori, kolom_jumlah, kolom_tombol = st.columns(
        [3.2, 1.5, 1.0, 1.2],
        vertical_alignment="bottom"
    )

    with kolom_cari:
        kata_kunci = st.text_input(
            "Cari resep atau bahan",
            placeholder="Contoh: ayam cabai, tempe kecap..."
        )

    with kolom_kategori:
        kategori = st.selectbox(
            "Kategori",
            ["Semua kategori"] + sorted(
                df["Category"].astype(str).unique().tolist()
            )
        )

    with kolom_jumlah:
        jumlah_hasil = st.selectbox(
            "Jumlah hasil",
            [5, 10, 15, 20],
            index=1
        )

    with kolom_tombol:
        cari = st.form_submit_button(
            "🔎 Cari resep",
            use_container_width=True
        )

if cari:
    if not kata_kunci.strip():
        st.warning("Masukkan nama masakan atau bahan terlebih dahulu.")
    else:
        query_vector = vectorizer.transform([kata_kunci.strip()])

        if query_vector.nnz == 0:
            st.warning(
                "Kata kunci tidak dikenali. Coba nama masakan "
                "atau bahan yang lain."
            )
        else:
            skor = cosine_similarity(query_vector, matrix).flatten()
            kandidat = df.copy()
            kandidat["_skor"] = skor

            if kategori != "Semua kategori":
                kandidat = kandidat[
                    kandidat["Category"].astype(str) == kategori
                ]

            hasil = (
                kandidat[kandidat["_skor"] > 0]
                .sort_values("_skor", ascending=False)
                .head(jumlah_hasil)
            )

            st.divider()
            st.subheader("Hasil Pencarian")
            st.write(
                f"Ditemukan **{len(hasil)} resep** untuk "
                f"kata kunci **{kata_kunci.strip()}**."
            )

            if hasil.empty:
                st.info(
                    "Belum ditemukan resep yang cocok. "
                    "Coba kata kunci atau kategori lain."
                )
            else:
                daftar_resep = list(hasil.iterrows())

                for awal in range(0, len(daftar_resep), 3):
                    baris = daftar_resep[awal:awal + 3]
                    kolom_kartu = st.columns(3, gap="medium")

                    for kolom, (indeks, resep) in zip(kolom_kartu, baris):
                        with kolom:
                            with st.container(border=True):
                                foto = cari_foto(
                                    resep["Category"], resep["Title"]
                                )
                                if foto:
                                    st.image(
                                        foto,
                                        use_container_width=True
                                    )

                                st.markdown(f"#### {resep['Title']}")
                                st.caption(
                                    f"🍽️ {resep['Category']}  ·  "
                                    f"Skor {resep['_skor']:.3f}"
                                )

                                with st.expander("Bahan & cara memasak"):
                                    st.markdown("**Bahan-bahan**")
                                    for bahan in tampilkan_daftar(
                                        resep["Ingredients"]
                                    ):
                                        st.markdown(f"- {bahan}")

                                    st.markdown("**Langkah memasak**")
                                    for nomor, langkah in enumerate(
                                        tampilkan_daftar(resep["Steps"]),
                                        start=1
                                    ):
                                        st.markdown(
                                            f"{nomor}. {langkah}"
                                        )

                                url = str(resep.get("URL", "")).strip()
                                if url.startswith(("https://", "http://")):
                                    st.link_button(
                                        "Lihat resep asli ↗",
                                        url,
                                        use_container_width=True
                                    )
else:
    st.markdown(
        "<div style='background:#eef4e9; padding:16px 20px; "
        "border-radius:14px; color:#31563b;'>"
        "💡 <b>Tips:</b> Coba cari dengan bahan yang ada di rumah, "
        "misalnya <i>tempe kecap</i>, <i>telur dadar</i>, atau "
        "<i>ayam pedas</i>.</div>",
        unsafe_allow_html=True
    )

st.divider()
st.caption(
    "DapurFinder · Sistem Temu Kembali Informasi Resep · "
    "Dikembangkan untuk proyek STKI"
)
