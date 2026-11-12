"""
Dashboard Prakiraan Kondisi Laut untuk Nelayan  (Streamlit, tampilan ramah HP)

Jalankan:   streamlit run dashboard_nelayan.py
Butuh   :   pip install streamlit plotly pandas numpy

Sumber data : 2 CSV prediksi holdout gabungan semua zona (SST = ARIMAX, Chl-a = XGBoost)
Kolom dipakai: zona_id, tanggal_terbit, horizon (1..4), target_date, y_pred (+ target, model bila ada)
Tampilan    : 3 peta (Potensi, SST, Chl-a) per WPP; pilih tanggal terbit lalu +1..+4 minggu.
"""
import glob
import math
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =====================================================================================
# 1. KONFIGURASI  (EDIT DI SINI)
# =====================================================================================
# 2 CSV prediksi holdout (gabungan semua zona). Boleh diisi path FILE .csv, atau path FOLDER:
# kalau folder, dipakai file "prediksi_holdout*.csv" di dalamnya (yang terbaru jika ada beberapa).
SST_PATH  = r"prediksi_holdout_2025_arimax_mingguan_FULL.csv"      # ARIMAX, SST
CHLA_PATH = r"prediksi_holdout_2025_xgboost_mingguan_log_FULL.csv"    # XGBoost, Chl-a

KKP_CSV = "Data KKP Hasil Tangkap Tahunan 2019-2024 - Copy.csv"

# Koordinat di zona_id (mis. "WPP712_-7.25_106.25") = titik "pusat" sel 0,5° atau "kiri_bawah"?
# CEK sekali: kalau semua angkanya berakhiran .25/.75 -> pusat; kalau .0/.5 -> kiri_bawah.
ZONA_TITIK = "pusat"
UKURAN_SEL = 0.5                 # derajat
COORD_CSV = None                 # opsional: CSV (zona_id, lat, lon) jika zona_id tidak memuat koordinat

HORIZON_MAKS = 4
NAMA_WPP = {
    712: "WPP-RI-712",
    713: "WPP-RI-713",
    718: "WPP-RI-718",
}

# =====================================================================================
# 2. AMBANG BATAS KLASIFIKASI POTENSI (HARDCODE)
# =====================================================================================
# Skor tiap variabel: 3 = optimum, 2 = toleransi, 1 = di luar.  Total = skor SST + skor Chl-a.
#   total 6      -> TINGGI   (SST dan Chl-a sama-sama optimum)
#   total 4 - 5  -> SEDANG
#   total 2 - 3  -> RENDAH
#
# BATAS OPTIMUM (dari paper, utamanya ikan pelagis kecil & TCT di perairan Indonesia):
#   SST 28-30 °C ; Chl-a 0,3-2,5 mg/m3   -> [R1] Prayitno dkk. 2021 (mengutip Rasyid 2010; Wicaksono 2017)
#   SST 29-30 °C ; Chl-a > 0,2 mg/m3     -> [R2] Kunarso 2005 (dikutip sumber sekunder)
#   SST >= 28,0 ; Chl-a > 0,22 mg/m3     -> [R7] "Standard Oceanographic Threshold" (jurnal ilmudata)
#   SST 29-30 ; Chl-a > 0,25 (TCT)       -> [R3] Balitbang KKP, Sulawesi Barat
#   Chl-a < 5 mg/m3 dianggap "aman"      -> [R9] Prihartanto 2009 dlm Prianto dkk. 2013 (via R1)
# BATAS TOLERANSI (27-28 & 30-31 °C ; 0,2-0,3 & 2,5-5 mg/m3) adalah KEPUTUSAN SAYA yang melonggarkan
# batas optimum satu "langkah" supaya ada kelas Sedang; BUKAN angka baku dari satu paper -> sesuaikan
# kalau pembimbing/kamu punya rujukan lain.
SST_OPTIMUM = (28.0, 30.0)
SST_TOLERANSI = (27.0, 31.0)
CHL_OPTIMUM = (0.3, 2.5)
CHL_TOLERANSI = (0.2, 5.0)

KELAS_NAMA = {0: "Tidak ada data", 1: "Rendah", 2: "Sedang", 3: "Tinggi"}
KELAS_WARNA = {0: "#bdbdbd", 1: "#e65100", 2: "#fdd835", 3: "#1b7837"}

REFERENSI = {
    "R1": "Prayitno, Rahman & Yasmi (2021), Aquatic 4(1) – optimum pelagis kecil SST 28–30 °C, Chl-a 0,3–2,5 mg/m³.",
    "R2": "Kunarso (2005), dikutip jurnal Buloma (Undip) & Manfish (Polnep) – Chl-a >0,2, SST 29–30 °C (sumber sekunder).",
    "R3": "Balitbang KKP, ZPPI tuna-cakalang-tongkol Sulawesi Barat – paling produktif SST 29–30 °C & Chl-a >0,25.",
    "R4": "Jurnal Fisheries (Unram, journal.unram.ac.id/jfh) – tangkapan cakalang tertinggi pada SST 28–29 °C.",
    "R5": "Paillin dkk., Jurnal Kelautan Tropis 23(2) – madidihang di Laut Seram: SST 29–30 °C potensial, <29 °C kurang potensial.",
    "R6": "Jurnal Manfish (Polnep), P. Lemukutan – teri potensial pada Chl-a >0,7 mg/m³ & SST 29–30 °C.",
    "R7": "Jurnal IJMST (ilmudata) – label potensial: SST ≥28,0 °C & Chl-a >0,22 mg/m³.",
    "R8": "Studi cakalang WPP 713 (core.ac.uk/347527686) – SST 29,0–29,5 °C, Chl-a 0,10–0,15 mg/m³.",
    "R9": "Prihartanto (2009) dalam Prianto dkk. (2013), via R1 – Chl-a <5 mg/m³ tergolong aman.",
}

# =====================================================================================
# 3. KISARAN OPTIMUM PER JENIS IKAN  (dicocokkan ke nama "Jenis Ikan" di data KKP)
# =====================================================================================
# (kata kunci, sst (lo,hi)|None, chl (lo,hi)|None, rujukan)   None pada lo/hi = tanpa batas
# Jenis ikan KKP yang TIDAK cocok dengan kata kunci mana pun (ikan dasar/karang, krustasea, dst.)
# tidak punya kisaran SST/Chl-a dari rujukan yang saya temukan -> TIDAK dipakai untuk menebak.
# Urutan penting: kata kunci yang lebih spesifik di atas.
SPESIES_DB = [
    (["TUNA MATA BESAR"], None, None, []),                                   # belum ada angka rujukan
    (["TUNA SIRIP KUNING", "MADIDIHANG"], (29.0, 30.0), None, ["R5"]),
    (["CAKALANG"], (28.0, 29.5), None, ["R4", "R8"]),                        # gabungan R4 (28-29) & R8 (29-29,5)
    (["TONGKOL"], (29.0, 30.0), (0.25, None), ["R3"]),
    (["TERI"], (29.0, 30.0), (0.7, None), ["R6"]),
    (["LAYANG", "KEMBUNG", "TEMBANG", "SELAR", "LEMURU", "SIRO", "BANYAR", "BENTONG"],
     (28.0, 30.0), (0.3, 2.5), ["R1", "R2"]),                                # kisaran umum pelagis kecil
]
KKP_TOP_N = 25          # jumlah jenis ikan teratas (volume 2019-2024) per WPP yang dipertimbangkan
MAKS_IKAN_HOVER = 3

# =====================================================================================
# 4. UTIL
# =====================================================================================
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
_RE_ZONA = re.compile(r"^\s*(?:WPP)?(\d{3})_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)\s*$", re.I)
_MAP = "map" if hasattr(go, "Choroplethmap") else "mapbox"
_CHORO = go.Choroplethmap if _MAP == "map" else go.Choroplethmapbox


def tgl(d) -> str:
    d = pd.Timestamp(d)
    return f"{d.day} {BULAN[d.month - 1]} {d.year}"


def num(x, nd=1) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}".replace(".", ",")


def rng_txt(r, nd=1) -> str:
    if r is None:
        return "—"
    lo, hi = r
    if lo is not None and hi is not None:
        return f"{num(lo, nd)}–{num(hi, nd)}"
    return f">{num(lo, 2)}" if lo is not None else f"<{num(hi, 2)}"


def koord_txt(lat, lon) -> str:
    return f"{abs(lat):.2f}°{'LS' if lat < 0 else 'LU'} {abs(lon):.2f}°{'BT' if lon >= 0 else 'BB'}".replace(".", ",")


def dalam(x, r) -> bool:
    if r is None:
        return True
    lo, hi = r
    return (lo is None or x >= lo) and (hi is None or x <= hi)


# =====================================================================================
# 5. LOAD DATA
# =====================================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve(path: str):
    """Path relatif dicoba dari folder kerja DAN dari folder file .py ini. Terima gaya Windows / Linux."""
    p = os.path.normpath(path.replace("\\", "/"))
    for c in (p, os.path.join(_HERE, p)):
        if os.path.exists(c):
            return c
    return None


def _cari_csv(path: str):
    p = _resolve(path)
    if p is None:
        return None
    if os.path.isfile(p):
        return p
    for pola in ("prediksi_holdout*.csv", "*.csv"):
        f = sorted(glob.glob(os.path.join(p, pola)), key=os.path.getmtime)
        if f:
            return f[-1]
    return None


@st.cache_data(show_spinner="Memuat prakiraan…")
def muat_prediksi(path: str, var: str, sig: float):
    kolom = {"zona_id", "tanggal_terbit", "horizon", "target_date", "y_pred", "target", "model"}
    df = pd.read_csv(path, usecols=lambda c: c in kolom, dtype={"zona_id": str})
    if "target" in df.columns:                                   # jaga-jaga kalau file memuat SST & Chl-a
        df = df[df["target"].astype(str).str.lower().str.contains(var)]
    model = str(df["model"].iloc[0]) if "model" in df.columns and len(df) else "?"
    if df["y_pred"].dtype == object:                             # desimal koma (ekspor Excel lokal)
        df["y_pred"] = df["y_pred"].astype(str).str.replace(",", ".", regex=False)
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
    df = df[(df["horizon"] >= 1) & (df["horizon"] <= HORIZON_MAKS)].copy()   # baris celah (<=0) dibuang
    df["horizon"] = df["horizon"].astype(int)
    for c in ("tanggal_terbit", "target_date"):                  # ISO atau format Excel bulan/tanggal/tahun
        df[c] = pd.to_datetime(df[c], format="mixed", dayfirst=False).dt.normalize()
    df = df.drop_duplicates(["zona_id", "tanggal_terbit", "horizon"], keep="last")
    return df[["zona_id", "tanggal_terbit", "horizon", "target_date", "y_pred"]].rename(columns={"y_pred": "nilai"}), model


@st.cache_data
def muat_zona(zona_ids: tuple, coord_csv) -> pd.DataFrame:
    rows = []
    ext = pd.read_csv(coord_csv, dtype={"zona_id": str}).set_index("zona_id") if coord_csv else None
    for z in zona_ids:
        m = _RE_ZONA.match(z)
        if m:
            rows.append((z, int(m.group(1)), float(m.group(2)), float(m.group(3))))
        elif ext is not None and z in ext.index:
            w = re.match(r"^\s*(?:WPP)?(\d{3})", z, re.I)
            rows.append((z, int(w.group(1)) if w else -1, float(ext.loc[z, "lat"]), float(ext.loc[z, "lon"])))
    return pd.DataFrame(rows, columns=["zona_id", "wpp", "lat", "lon"])


@st.cache_data
def buat_geojson(zona: pd.DataFrame) -> dict:
    h = UKURAN_SEL / 2
    feats = []
    for z, lat, lon in zona[["zona_id", "lat", "lon"]].itertuples(index=False):
        s, w = (lat - h, lon - h) if ZONA_TITIK == "pusat" else (lat, lon)
        n, e = s + UKURAN_SEL, w + UKURAN_SEL
        feats.append({"type": "Feature", "id": z, "properties": {},
                      "geometry": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}})
    return {"type": "FeatureCollection", "features": feats}


@st.cache_data
def muat_kkp(path: str) -> dict:
    """wpp -> daftar kandidat ikan (punya kisaran SST/Chl-a dari rujukan) urut volume tangkapan."""
    path = _resolve(path)
    if path is None:
        return {}
    d = pd.read_csv(path, sep=None, engine="python")
    d["wpp"] = pd.to_numeric(d["WPP"].astype(str).str.extract(r"(\d{3})")[0], errors="coerce")
    d["ton"] = pd.to_numeric(d["Volume Produksi (Ton)"], errors="coerce")
    d = d[d["wpp"].notna() & ~d["Jenis Ikan"].astype(str).str.upper().str.contains("LAINNYA")]
    g = d.groupby(["wpp", "Jenis Ikan"])["ton"].sum().reset_index().sort_values("ton", ascending=False)
    out = {}
    for wpp, gg in g.groupby("wpp"):
        cands = []
        for nama, ton in gg.head(KKP_TOP_N)[["Jenis Ikan", "ton"]].itertuples(index=False):
            up = str(nama).upper()
            for kata, sst, chl, ref in SPESIES_DB:
                if any(re.search(rf"\b{re.escape(k)}\b", up) for k in kata):
                    if sst is not None or chl is not None:
                        cands.append(dict(nama=str(nama).title().replace("; ", " / "), ton=float(ton),
                                          sst=sst, chl=chl, ref=ref))
                    break
        out[int(wpp)] = cands
    return out


# =====================================================================================
# 6. KLASIFIKASI & HOVER
# =====================================================================================
def _skor(x, opt, tol):
    x = np.asarray(x, float)
    s = np.where((x >= opt[0]) & (x <= opt[1]), 3, np.where((x >= tol[0]) & (x <= tol[1]), 2, 1))
    return np.where(np.isnan(x), 0, s)


def kelas_potensi(sst, chl):
    s, c = _skor(sst, SST_OPTIMUM, SST_TOLERANSI), _skor(chl, CHL_OPTIMUM, CHL_TOLERANSI)
    tot = s + c
    return np.where((s == 0) | (c == 0), 0, np.where(tot >= 6, 3, np.where(tot >= 4, 2, 1)))


def ikan_cocok(kkp, wpp, sst, chl):
    if np.isnan(sst) or np.isnan(chl):
        return []
    return [c for c in kkp.get(int(wpp), []) if dalam(sst, c["sst"]) and dalam(chl, c["chl"])][:MAKS_IKAN_HOVER]


def susun_hover(r, kkp, fokus: str) -> str:
    baris = [f"<b>WPP {int(r.wpp)} · {koord_txt(r.lat, r.lon)}</b>"]
    if fokus == "potensi":
        baris.append(f"Potensi: <b>{KELAS_NAMA[int(r.kelas)]}</b>")
    baris.append(f"SST {num(r.sst, 1)} °C · Klorofil-a {num(r.chl, 2)} mg/m³")
    if int(r.wpp) not in kkp:
        baris.append("<i>Data KKP belum dimuat</i>" if not kkp else "<i>Tidak ada data KKP untuk WPP ini</i>")
        return "<br>".join(baris)
    cocok = ikan_cocok(kkp, r.wpp, r.sst, r.chl)
    if cocok:
        baris.append("Ikan yang mungkin ada (tangkapan KKP):")
        baris += [f"• {c['nama']} (SST {rng_txt(c['sst'])} °C; Klorofil-a {rng_txt(c['chl'], 2)})" for c in cocok]
    elif not (np.isnan(r.sst) or np.isnan(r.chl)):
        baris.append("<i>Belum ada jenis ikan utama WPP ini yang cocok dengan kondisi ini</i>")
    baris.append("<i>Perkiraan, bukan pengamatan di titik ini</i>")
    return "<br>".join(baris)


# =====================================================================================
# 7. PETA
# =====================================================================================
def _view(zona_w):
    lat0, lat1, lon0, lon1 = zona_w.lat.min(), zona_w.lat.max(), zona_w.lon.min(), zona_w.lon.max()
    span_lon, span_lat = max(lon1 - lon0 + 1.5, 1.5), max(lat1 - lat0 + 1.5, 1.5)
    z = min(math.log2(360 * 360 / (256 * span_lon)), math.log2(360 * 460 / (256 * span_lat * 1.15))) - 0.2
    return dict(lat=(lat0 + lat1) / 2, lon=(lon0 + lon1) / 2), float(np.clip(z, 3, 9))


def buat_peta(df, geo, jenis, zona_w, kkp, rng_sst):
    centre, zoom = _view(zona_w)
    df = df.copy()
    df["hover"] = [susun_hover(r, kkp, jenis) for r in df.itertuples()]
    common = dict(geojson=geo, locations=df["zona_id"], text=df["hover"],
                  hovertemplate="%{text}<extra></extra>",
                  marker=dict(opacity=0.8, line=dict(width=0.4, color="rgba(255,255,255,0.7)")))
    if jenis == "potensi":
        k = df["kelas"].astype(int)
        w = [KELAS_WARNA[i] for i in range(4)]
        sk = [[0, w[0]], [.25, w[0]], [.25, w[1]], [.5, w[1]], [.5, w[2]], [.75, w[2]], [.75, w[3]], [1, w[3]]]
        tr = _CHORO(z=k, zmin=-0.5, zmax=3.5, colorscale=sk, showscale=False, **common)
        h = 480
    else:
        if jenis == "sst":
            z, zmin, zmax, ttl = df["sst"], rng_sst[0], rng_sst[1], "Suhu permukaan laut (°C)"
            cb = dict(title=dict(text=ttl, side="top"))
        else:
            lo, hi = 0.05, 5.0
            z = np.log10(df["chl"].clip(lo, hi))
            zmin, zmax = math.log10(lo), math.log10(hi)
            ticks = [0.1, 0.3, 1, 3]
            cb = dict(title=dict(text="Klorofil-a (mg/m³)", side="top"),
                      tickvals=[math.log10(t) for t in ticks], ticktext=[str(t).replace(".", ",") for t in ticks])
        # Turbo: biru gelap (dingin/rendah) -> cyan -> hijau -> kuning -> oranye -> merah gelap (panas/tinggi)
        tr = _CHORO(z=z, zmin=zmin, zmax=zmax, colorscale="Turbo", showscale=True,
                    colorbar=dict(orientation="h", x=0.5, xanchor="center", y=-0.03, yanchor="top",
                                  len=0.92, thickness=12, **cb), **common)
        h = 540
    fig = go.Figure(tr)
    fig.update_layout(**{_MAP: dict(style="carto-positron", center=centre, zoom=zoom)},
                      margin=dict(l=0, r=0, t=0, b=34 if jenis == "potensi" else 90),
                      height=h, dragmode="pan", hoverlabel=dict(font_size=13, align="left"))
    return fig


def tampil(fig):
    cfg = {"displayModeBar": False, "scrollZoom": False}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg, theme=None)
    except TypeError:                                         # Streamlit versi lama
        st.plotly_chart(fig, use_container_width=True, config=cfg, theme=None)


# =====================================================================================
# 8. APLIKASI
# =====================================================================================
CSS = """
<style>
.block-container {padding: .8rem .8rem 2.5rem .8rem; max-width: 760px;}
h1 {font-size: 1.55rem !important; margin-bottom: .1rem;}
.chip {display:inline-block; padding: 2px 11px; margin: 0 6px 6px 0; border-radius: 999px;
       font-size: .85rem; font-weight: 600; color:#111;}
div[role="radiogroup"] label {font-size: 1.02rem;}
footer {visibility: hidden;}
</style>
"""


def main():
    st.set_page_config(page_title="Prakiraan Laut untuk Nelayan", page_icon="🎣",
                       layout="centered", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("🎣 Prakiraan Kondisi Laut")
    st.caption("Perkiraan potensi daerah tangkapan 1–4 minggu ke depan dari suhu laut (SST) dan klorofil-a.")

    # ---- data ----
    data, model_dipakai = {}, {}
    for var, p in (("sst", SST_PATH), ("chla", CHLA_PATH)):
        f = _cari_csv(p)
        if not f:
            st.error(f"File prediksi holdout untuk '{var}' tidak ditemukan di: {p}. "
                     f"Folder kerja: {os.getcwd()} · folder skrip: {_HERE}. "
                     f"Cek SST_PATH / CHLA_PATH di bagian atas dashboard_nelayan.py.")
            st.stop()
        data[var], model_dipakai[var] = muat_prediksi(f, var, os.path.getmtime(f))

    zona = muat_zona(tuple(sorted(set(data["sst"].zona_id) | set(data["chla"].zona_id))), COORD_CSV)
    if zona.empty:
        st.error("zona_id tidak memuat koordinat (format WPP712_-7.25_106.25). Isi COORD_CSV (zona_id, lat, lon).")
        st.stop()
    geo = buat_geojson(zona)
    kkp = muat_kkp(KKP_CSV)
    if not kkp:
        st.warning(f"Data KKP tidak terbaca ({KKP_CSV}); perkiraan jenis ikan tidak akan muncul. "
                   f"Letakkan CSV di folder kerja ({os.getcwd()}) atau folder skrip ({_HERE}), atau ubah KKP_CSV.")

    sst = data["sst"].rename(columns={"nilai": "sst"})
    chl = data["chla"].rename(columns={"nilai": "chl", "target_date": "target_date_c"})
    allp = sst.merge(chl, on=["zona_id", "tanggal_terbit", "horizon"], how="outer")
    allp["target_date"] = allp["target_date"].fillna(allp["target_date_c"])

    # ---- kontrol (di halaman, bukan sidebar, supaya mudah di HP) ----
    wpp_ada = sorted(zona.wpp.unique())
    wpp_sel = st.selectbox("Wilayah", wpp_ada, format_func=lambda w: NAMA_WPP.get(w, f"WPP {w}"))
    tanggal = sorted(allp["tanggal_terbit"].dropna().unique(), reverse=True)

    # Horizon apa saja yang tersedia untuk tiap tanggal terbit. Di data holdout, minggu terbit paling
    # baru biasanya cuma punya +1 minggu (target +2..+4 sudah lewat akhir data), jadi default
    # dipilih minggu terbaru yang horizon +1..+4-nya LENGKAP.
    cover = {pd.Timestamp(t): sorted(int(h) for h in g["horizon"].dropna().unique())
             for t, g in allp.dropna(subset=["tanggal_terbit"]).groupby("tanggal_terbit")}
    idx0 = next((i for i, d in enumerate(tanggal) if len(cover.get(pd.Timestamp(d), [])) >= HORIZON_MAKS), 0)

    def _label_tanggal(d):
        hs = cover.get(pd.Timestamp(d), [])
        extra = "" if len(hs) >= HORIZON_MAKS else f" (hanya {', '.join(f'+{h}' for h in hs)} minggu)"
        return f"Minggu s.d. {tgl(d)}{extra}"

    t_sel = st.selectbox("Minggu terbit prakiraan", tanggal, index=idx0, format_func=_label_tanggal)
    h_ada = cover.get(pd.Timestamp(t_sel), [])
    if not h_ada:
        st.warning("Belum ada prakiraan untuk minggu terbit ini.")
        st.stop()
    h_sel = st.radio("Prakiraan untuk", h_ada, horizontal=True, format_func=lambda h: f"+{h} minggu")

    sub = allp[(allp.tanggal_terbit == t_sel) & (allp.horizon == h_sel)].merge(zona, on="zona_id")
    sub = sub[sub.wpp == wpp_sel].reset_index(drop=True)
    if sub.empty:
        st.warning("Belum ada prakiraan untuk pilihan ini.")
        st.stop()
    t_target = sub["target_date"].dropna().iloc[0] if sub["target_date"].notna().any() else None
    if t_target is not None:
        st.info(f"Prakiraan untuk **minggu s.d. {tgl(t_target)}** (diterbitkan minggu s.d. {tgl(t_sel)}).")

    sub["kelas"] = kelas_potensi(sub["sst"], sub["chl"])
    zona_w = zona[zona.wpp == wpp_sel]
    lo, hi = np.nanpercentile(data["sst"]["nilai"], [1, 99])
    rng_sst = (math.floor(lo * 2) / 2, math.ceil(hi * 2) / 2)

    # ---- ringkasan ----
    n = len(sub)
    cnt = sub["kelas"].value_counts()
    chips = "".join(f'<span class="chip" style="background:{KELAS_WARNA[k]}">'
                    f'{KELAS_NAMA[k]}: {int(cnt.get(k, 0))} zona ({cnt.get(k, 0) / n:.0%})</span>' for k in (3, 2, 1))
    st.markdown(chips, unsafe_allow_html=True)

    # ---- peta ----
    tab_p, tab_s, tab_c = st.tabs(["🎣 Potensi", "🌡️ Suhu laut", "🌿 Klorofil-a"])
    with tab_p:
        legenda = "".join(f'<span class="chip" style="background:{KELAS_WARNA[k]}">{KELAS_NAMA[k]}</span>'
                          for k in (3, 2, 1, 0))
        st.markdown(legenda, unsafe_allow_html=True)
        tampil(buat_peta(sub, geo, "potensi", zona_w, kkp, rng_sst))
        st.caption("Ketuk sebuah kotak untuk melihat detail dan perkiraan jenis ikan.")
        top = sub[sub.kelas == 3].sort_values("chl", ascending=False).head(15)
        if len(top):
            with st.expander(f"Daftar zona potensi tinggi ({int((sub.kelas == 3).sum())} zona, 15 teratas)"):
                st.dataframe(pd.DataFrame({
                    "Koordinat": [koord_txt(a, b) for a, b in zip(top.lat, top.lon)],
                    "SST (°C)": top.sst.round(1), "Klorofil-a (mg/m³)": top.chl.round(2)}), hide_index=True)
    with tab_s:
        tampil(buat_peta(sub, geo, "sst", zona_w, kkp, rng_sst))
        st.caption("Merah = hangat · Biru = dingin. Warna makin gelap = mendekati ujung skala.")
    with tab_c:
        tampil(buat_peta(sub, geo, "chla", zona_w, kkp, rng_sst))
        st.caption("Merah = klorofil-a tinggi (perairan subur) · Biru = rendah.")

    # ---- cek data ----
    with st.expander("Cek data (jumlah baris per horizon)"):
        cek = pd.DataFrame({"SST": data["sst"].groupby("horizon").size(),
                            "Klorofil-a": data["chla"].groupby("horizon").size()})
        cek = cek.reindex(range(1, HORIZON_MAKS + 1)).fillna(0).astype(int)
        cek.index = [f"+{h} minggu" for h in cek.index]
        st.dataframe(cek)
        st.caption("Kalau +2 s.d. +4 minggu bernilai 0, file CSV memang belum memuat horizon itu "
                   "(perlu diekspor ulang dari notebook model).")

    # ---- keterangan ----
    with st.expander("Tentang peta ini"):
        st.markdown(
            "**Cara klasifikasi potensi** (skor SST + skor Klorofil-a, dari ambang pada paper):\n\n"
            f"- Optimum: SST {rng_txt(SST_OPTIMUM)} °C dan Klorofil-a {rng_txt(CHL_OPTIMUM, 1)} mg/m³ → skor 3\n"
            f"- Toleransi: SST {rng_txt(SST_TOLERANSI)} °C dan Klorofil-a {rng_txt(CHL_TOLERANSI, 1)} mg/m³ → skor 2; di luar itu → 1\n"
            "- Tinggi = kedua variabel optimum · Sedang = salah satu hanya toleransi · Rendah = selebihnya\n\n"
            "**Perkiraan jenis ikan**: jenis ikan dengan tangkapan terbesar di WPP tersebut (KKP 2019–2024) "
            "yang kisaran SST/Klorofil-a optimumnya cocok dengan kondisi sel. Ini perkiraan, bukan jaminan ada ikan; "
            "ikan dasar/karang dan krustasea tidak ditebak karena tidak ada kisaran SST/Klorofil-a dari rujukan.\n\n"
            f"**Model**: SST = {model_dipakai['sst']}, Klorofil-a = {model_dipakai['chla']} (prediksi holdout 2025).")
        st.markdown("**Rujukan**")
        for k, v in REFERENSI.items():
            st.markdown(f"- [{k}] {v}")


if __name__ == "__main__":
    main()
