import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
import textwrap
import re
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyBboxPatch

BASE_DIR = r"C:/Users/eyada/OneDrive/Desktop/Praxis"
CSV_PATH = os.path.join(BASE_DIR, "predictions_results", "results_v5.csv")
HM_DIR = os.path.join(BASE_DIR, "predictions_results", "heatmaps_v5")
TS = datetime.now().strftime('%Y%m%d_%H%M%S')
PDF_PATH = os.path.join(BASE_DIR, "predictions_results", f"final_bericht_{TS}.pdf")

INSTITUTION = "Technische Hochschule Brandenburg"
AUTHOR_NAME = "Eiad Aljazairli"
PROJECT_NOTE = "Satellitenbilder – Vorher/Nachher – Twin-EfficientNetB0 – Regression (kalibriert)"
LOGO_PATH = os.path.join(BASE_DIR, "THB-logo.webp")

ABSTRACT_TEXT = (
    "Wir schätzen Gebäudeschäden automatisch aus Satellitenbildern. Aus Vorher/Nachher-Patches (256x256) "
    "berechnet ein EfficientNet-B0 Prozentwerte. Diese werden als Heatmap, Kennzahlen und Top-10-Beispiele "
    "dargestellt. Der Ablauf ist schnell, reproduzierbar und auf andere Gebiete übertragbar."
)

PATCH_SIZE = 256
FIGSIZE_A4 = (11.69, 8.27)
DPI = 300
EXTS = [".jpg", ".jpeg", ".png", ".webp"]
FORCE_LAST_KEYS = ["256_1280_768"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.titlesize": 16,
    "pdf.fonttype": 42,
    "savefig.dpi": DPI
})
BG = "#FFFFFF"
INK = "#0F172A"
MUTED = "#475569"
HAIR = "#E5E7EB"

def wrap_text(s: str, width: int = 110) -> str:
    return "\n".join(textwrap.fill(s, width=width, break_long_words=False, break_on_hyphens=False).splitlines())

def short_path(p: str, keep_parts: int = 2) -> str:
    if not p:
        return ""
    parts = Path(p).parts
    tail = parts[-keep_parts:] if len(parts) > keep_parts else parts
    return "…/" + "/".join(tail)

def read_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = {"name", "before_path", "after_path"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV muss Spalten enthalten: {need}. Gefunden: {list(df.columns)}")
    df = df.copy()
    if "pred_calibrated_percent" in df.columns:
        x = pd.to_numeric(df["pred_calibrated_percent"], errors="coerce")
    elif "percent" in df.columns:
        x = pd.to_numeric(df["percent"], errors="coerce")
    else:
        raise ValueError("CSV benötigt Spalte 'percent' oder 'pred_calibrated_percent'.")
    mx = np.nanmax(x.to_numpy())
    df["pred_cal"] = x.astype(float) if (np.isfinite(mx) and mx <= 1.5) else x.astype(float) / 100.0
    if "is_damaged" in df.columns:
        df["is_damaged"] = df["is_damaged"].astype(str).str.strip().str.lower().isin(["1","true","t","yes","y"])
    scenes = []
    for p in df["after_path"].astype(str).tolist():
        try:
            scenes.append(Path(p).parent.name)
        except Exception:
            scenes.append("scene")
    df["scene"] = scenes
    return df

def basic_stats(series: pd.Series) -> dict:
    arr = series.dropna().to_numpy()
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "median": float(np.median(arr)) if arr.size else float("nan"),
        "std": float(np.std(arr)) if arr.size else float("nan"),
        "min": float(np.min(arr)) if arr.size else float("nan"),
        "max": float(np.max(arr)) if arr.size else float("nan"),
        "p95": float(np.percentile(arr, 95)) if arr.size else float("nan"),
        "p99": float(np.percentile(arr, 99)) if arr.size else float("nan"),
    }

def _flatten_rgba_on_white(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, (0, 0), img.split()[-1])
        return bg.convert("RGB")
    return img.convert("RGB")

def _autocrop_edges(img: Image.Image, tol_black: int = 3, tol_white: int = 252) -> Image.Image:
    arr = np.array(img)
    if img.mode in ("RGBA", "LA"):
        alpha = np.array(img.getchannel("A"))
        mask = alpha > 0
    else:
        gray = arr.mean(axis=2).astype(np.uint8) if arr.ndim == 3 else arr
        mask = (gray > tol_black) & (gray < tol_white)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size and cols.size:
        r0, r1 = int(rows[0]), int(rows[-1])
        c0, c1 = int(cols[0]), int(cols[-1])
        img = img.crop((c0, r0, c1 + 1, r1 + 1))
    return img

def open_image_clean(path: str) -> Image.Image:
    try:
        img = Image.open(path)
        img = _autocrop_edges(img)
        img = _flatten_rgba_on_white(img)
        img = _autocrop_edges(img)
        return img
    except Exception:
        return Image.new("RGB", (PATCH_SIZE, PATCH_SIZE), (230, 230, 230))

def _fit_axes_for_image(fig, img_w, img_h, left=0.03, right=0.99, bottom=0.16, top=0.88):
    box_w = right - left
    box_h = top - bottom
    aspect = img_w / img_h
    if box_w / box_h > aspect:
        w = box_h * aspect; h = box_h
    else:
        w = box_w; h = box_w / aspect
    x = left + (box_w - w) / 2
    y = bottom + (box_h - h) / 2
    return fig.add_axes([x, y, w, h])

def parse_after_before(path_str: str) -> Optional[Tuple[Path, str, int, int, int, str]]:
    s = str(path_str)
    m = re.search(r'(.*[\\/])?(after|before)(\d+)_(-?\d+)_(-?\d+)\.(jpg|jpeg|png|webp)$', s, re.IGNORECASE)
    if not m:
        return None
    dirpart = Path(m.group(1)) if m.group(1) else Path(".")
    tag = m.group(2).lower()
    size = int(m.group(3))
    x = int(m.group(4))
    y = int(m.group(5))
    ext = "." + m.group(6).lower()
    return dirpart, tag, size, x, y, ext

def make_path(dirpart: Path, tag: str, size: int, x: int, y: int, ext: str) -> Path:
    return dirpart / f"{tag}{size}_{x}_{y}{ext}"

def find_counterpart_by_name(p: str, desired_tag: str) -> Optional[str]:
    info = parse_after_before(p)
    if not info:
        return None
    dirpart, _, size, x, y, ext = info
    candidates = []
    for ext_try in [ext] + [e for e in EXTS if e != ext]:
        for (xx, yy) in [(x, y), (y, x)]:
            cand = make_path(dirpart, desired_tag, size, xx, yy, ext_try)
            if cand.exists():
                candidates.append(cand)
    if candidates:
        return str(candidates[0])
    return None

def canonical_pair(before_path: str, after_path: str) -> Tuple[str, str, str]:
    b = str(before_path) if isinstance(before_path, str) else ""
    a = str(after_path) if isinstance(after_path, str) else ""
    if a:
        alt_b = find_counterpart_by_name(a, "before")
        if alt_b:
            b = alt_b
    if not a and b:
        alt_a = find_counterpart_by_name(b, "after")
        if alt_a:
            a = alt_a
    key = ""
    info = parse_after_before(a) or parse_after_before(b)
    if info:
        _, _, size, x, y, _ = info
        key = f"{size}_{x}_{y}"
    return b, a, key

def add_fixed_and_keys(df: pd.DataFrame) -> pd.DataFrame:
    bf, af, keys = [], [], []
    for _, r in df.iterrows():
        b, a, k = canonical_pair(str(r.get("before_path", "")), str(r.get("after_path", "")))
        bf.append(b); af.append(a); keys.append(k if k else str(r.get("name", "")))
    df2 = df.copy()
    df2["before_fixed"] = bf
    df2["after_fixed"] = af
    df2["tile_key"] = keys
    return df2

def fig_cover() -> plt.Figure:
    fig = plt.figure(figsize=FIGSIZE_A4, facecolor=BG)
    fig.subplots_adjust(left=0.06, right=0.95, top=0.92, bottom=0.10)
    y = 0.92
    fig.text(0.05, y, INSTITUTION, fontsize=30, fontweight="bold", color=INK); y -= 0.08
    fig.text(0.05, y, "Automatisierte Klassifikation von Gebäudeschäden", fontsize=15, color=INK); y -= 0.045
    fig.text(0.05, y, "mittels Machine Learning auf Grundlage von", fontsize=15, color=INK); y -= 0.040
    fig.text(0.05, y, "Satellitenbildern am Beispiel von Damaskus", fontsize=15, color=INK); y -= 0.055
    meta = [
        f"Autor: {AUTHOR_NAME}",
        f"Datum: {datetime.now().strftime('%d.%m.%Y')}",
        f"Projekt: {PROJECT_NOTE}",
        "Studiengang: Informatik (B.Sc.) – Bachelorarbeit",
    ]
    for m in meta:
        fig.text(0.05, y, m, fontsize=12.5, color=MUTED); y -= 0.028
    y -= 0.05
    fig.text(0.05, y, "Abstract", fontsize=15, fontweight="bold", color=INK); y -= 0.045
    box = dict(boxstyle="round,pad=0.5,rounding_size=6", fc="white", ec=HAIR, lw=1.0)
    fig.text(0.05, y, wrap_text(ABSTRACT_TEXT, 110), fontsize=11.5, color=INK, bbox=box)
    try:
        p = Path(LOGO_PATH)
        if p.exists():
            logo = Image.open(p).convert("RGBA")
            max_w = int(FIGSIZE_A4[0] * DPI * 0.18)
            sc = min(1.0, max_w / logo.width)
            logo = logo.resize((int(logo.width * sc), int(logo.height * sc)))
            fig_w, fig_h = int(FIGSIZE_A4[0]*DPI), int(FIGSIZE_A4[1]*DPI)
            xo = fig_w - logo.width - int(0.055 * fig_w)
            yo = int(0.075 * fig_h)
            fig.figimage(logo, xo=xo, yo=yo)
    except Exception:
        pass
    return fig

def fig_stats(df: pd.DataFrame, title: str, caption: str) -> plt.Figure:
    st = basic_stats(df["pred_cal"])
    fig = plt.figure(figsize=FIGSIZE_A4, facecolor=BG); ax = plt.gca(); ax.axis("off")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.12)
    lines = [
        title, "",
        f"Anzahl Patches: {st['count']}",
        f"Durchschnitt:   {st['mean']*100:.2f}%",
        f"Median:         {st['median']*100:.2f}%",
        f"Std-Abw.:       {st['std']*100:.2f}%",
        f"Minimum:        {st['min']*100:.2f}%",
        f"Maximum:        {st['max']*100:.2f}%",
        f"95. Perzentil:  {st['p95']*100:.2f}%",
        f"99. Perzentil:  {st['p99']*100:.2f}%",
    ]
    ax.text(0.05, 0.90, "\n".join(lines), va="top", ha="left", fontsize=14, color=INK)
    ax.text(0.05, 0.10, wrap_text(caption, 140), va="bottom", ha="left", fontsize=10.5, color=MUTED)
    return fig

def fig_bullets(title: str, items: List[str], footnote: str = "") -> plt.Figure:
    fig = plt.figure(figsize=FIGSIZE_A4, facecolor=BG); ax = plt.gca(); ax.axis("off")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.10)
    fig.text(0.05, 0.90, title, fontsize=16, fontweight="bold", ha="left", va="top", color=INK)
    bullets = []
    for it in items:
        lines = textwrap.wrap(it, width=118, break_long_words=False, break_on_hyphens=False)
        if not lines:
            continue
        bullets.append("• " + lines[0])
        for ln in lines[1:]:
            bullets.append("  " + ln)
        bullets.append("")
    fig.text(0.05, 0.85, "\n".join(bullets).rstrip(), fontsize=12.2, ha="left", va="top", color=INK)
    if footnote:
        fig.text(0.05, 0.10, wrap_text(footnote, 140), fontsize=10.5, ha="left", va="bottom", color=MUTED)
    return fig

def fig_full_image_with_caption(img_path: str, title: str, caption: str) -> plt.Figure:
    img = open_image_clean(img_path)
    fig = plt.figure(figsize=FIGSIZE_A4, facecolor=BG)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.90, bottom=0.12)
    ax_img = _fit_axes_for_image(fig, img.width, img.height, left=0.04, right=0.98, bottom=0.20, top=0.88)
    ax_img.imshow(img); ax_img.axis("off")
    ax_img.add_patch(FancyBboxPatch((0,0),1,1, transform=ax_img.transAxes, boxstyle="round,pad=0.004,rounding_size=6", fc="none", ec=HAIR, lw=0.9))
    fig.text(0.5, 0.905, title, fontsize=15, ha="center", va="top", color=INK)
    fig.text(0.05, 0.06, wrap_text(caption, 150), ha="left", va="bottom", fontsize=10.5, color=MUTED)
    return fig

def chunked(iterable, n: int) -> List[list]:
    it = list(iterable)
    return [it[i:i+n] for i in range(0, len(it), n)]

def build_top_k_with_forced_end(df_fixed: pd.DataFrame, k: int, force_last_keys: List[str]) -> pd.DataFrame:
    base = df_fixed.sort_values(["pred_cal","tile_key"], ascending=[False, True])
    base_wo_forced = base[~base["tile_key"].isin(force_last_keys)]
    top_keep = base_wo_forced.head(max(0, k - len(force_last_keys))).copy()
    forced_rows = df_fixed[df_fixed["tile_key"].isin(force_last_keys)].drop_duplicates(subset=["tile_key"])
    sel = pd.concat([top_keep, forced_rows], ignore_index=True)
    sel = sel.drop_duplicates(subset=["tile_key"]).head(k).reset_index(drop=True)
    return sel

def add_card(ax):
    r = FancyBboxPatch((0.0, 0.0), 1.0, 1.0, transform=ax.transAxes, boxstyle="round,pad=0.016,rounding_size=8", fc="#FFFFFF", ec=HAIR, lw=0.8, zorder=-1)
    ax.add_patch(r)

def fig_top_gallery_pages_with_before_after(df: pd.DataFrame, k: int, title: str, caption: str, force_last_keys: Optional[List[str]] = None) -> List[plt.Figure]:
    df_fixed = add_fixed_and_keys(df)
    sel = build_top_k_with_forced_end(df_fixed, k, force_last_keys) if force_last_keys else df_fixed.sort_values(["pred_cal","tile_key"], ascending=[False,True]).head(k).copy()
    sel["pct"] = (sel["pred_cal"] * 100).round(2)
    rows_per_page = 6
    pages: List[plt.Figure] = []
    rows = list(sel.iterrows())
    for page_rows in [rows[i:i+rows_per_page] for i in range(0, len(rows), rows_per_page)]:
        fig = plt.figure(figsize=FIGSIZE_A4, facecolor=BG)
        fig.subplots_adjust(left=0.05, right=0.99, top=0.90, bottom=0.10)
        gs = GridSpec(len(page_rows), 3, figure=fig, width_ratios=[1.05, 1.05, 2.1], hspace=0.55, wspace=0.35)
        fig.suptitle(title, fontsize=15, color=INK)
        for i, (_, r) in enumerate(page_rows):
            ax_b = fig.add_subplot(gs[i, 0]); ax_b.axis('off')
            p_b = r.get('before_fixed')
            b_img = open_image_clean(str(p_b)) if p_b and Path(str(p_b)).exists() else Image.new('RGB', (PATCH_SIZE, PATCH_SIZE), (230,230,230))
            ax_b.imshow(b_img); ax_b.set_title('Before', fontsize=9, color=MUTED)
            add_card(ax_b)

            ax_a = fig.add_subplot(gs[i, 1]); ax_a.axis('off')
            p_a = r.get('after_fixed')
            a_img = open_image_clean(str(p_a)) if p_a and Path(str(p_a)).exists() else Image.new('RGB', (PATCH_SIZE, PATCH_SIZE), (230,230,230))
            ax_a.imshow(a_img); ax_a.set_title('After', fontsize=9, color=MUTED)
            add_card(ax_a)

            ax_t = fig.add_subplot(gs[i, 2]); ax_t.axis('off')
            add_card(ax_t)
            lines = [
                f"Name: {r.get('tile_key', r.get('name',''))}",
                f"Zerstörung: {r['pct']:.2f}%",
                f"Before: {short_path(r.get('before_fixed',''))}",
                f"After:  {short_path(r.get('after_fixed',''))}",
            ]
            ax_t.text(0.04, 0.86, "\n".join(lines), va='top', ha='left', fontsize=10.8, color=INK, transform=ax_t.transAxes)
        fig.text(0.05, 0.06, wrap_text(caption, 150), ha='left', va='bottom', fontsize=10.5, color=MUTED)
        pages.append(fig)
    return pages

def stamp_page(fig, n):
    fig.text(0.97, 0.03, f"Seite {n}", ha="right", va="bottom", fontsize=9.5, color=MUTED)

def run(pdf_path: str, csv_path: str, hm_dir: str, topk: int):
    df_all = read_results(csv_path)
    after_img_path = os.path.join(hm_dir, "after_mosaic.jpg")
    heat_img_path = os.path.join(hm_dir, "heatmap_colored.png")
    colorbar_path = os.path.join(hm_dir, "colorbar.png")
    cap_stats = "Hinweis: Alle Prozentwerte beziehen sich auf die Modellvorhersagen pro Patch (0–100 %)."
    caption_after = "Mosaik aus allen Nachher-Patches für den betrachteten Ausschnitt."
    caption_heat = "Schadensverteilung als kontinuierliche Heatmap. Intensivere Töne entsprechen höheren Prozentwerten."
    caption_cbar = "Farbskala der Prozentwerte (0–100 %): je weiter nach rechts, desto höher die vermutete Zerstörung."
    caption_top = "Top-Patches mit jeweils Vorher/Nachher und geschätztem Schaden."
    with PdfPages(pdf_path) as pdf:
        try:
            info = pdf.infodict()
            info["Title"] = "final_bericht"
            info["Author"] = AUTHOR_NAME
            info["Subject"] = "Damage estimation report"
        except Exception:
            pass
        fig = fig_cover(); pdf.savefig(fig, dpi=DPI); plt.close(fig)
        page = 2
        fig = fig_stats(df_all, "Kennzahlen (gesamt)", cap_stats)
        stamp_page(fig, page); page += 1
        pdf.savefig(fig, dpi=DPI); plt.close(fig)
        bullets = [
            "Eingabe: Vorher/Nachher-Patches (256×256) pro Ort; radiometrisch normalisiert.",
            "Modell: Twin-EfficientNet-B0 mit geteilten Gewichten; Fusion der Merkmalsvektoren und Regressor für einen Schadenswert von 0–100 %.",
            "Genutzte Bildhinweise: Veränderungen an Gebäude-Kanten und Grundrissen, Brüche an Dachkanten, fehlende Ecken/Wände.",
            "Trümmer- und Schutttexturen (feinkörnige, chaotische Muster), Aufrauung von Flächen.",
            "Dachöffnungen/fehlende Dächer und veränderte Schattengeometrie; Verdunkelungen in Innenräumen.",
            "Verschwundene/verschobene Gebäude-Footprints, Helligkeits- und Farbtonwechsel (Staub/Brandspuren).",
            "Kalibrierung/Skalierung: Abbildung der Rohwerte auf Prozentwerte anhand der Validierung.",
            "Heatmap: Raster der Patch-Vorhersagen, auf Fläche gelegt; Farbskala 0–100 %.",
            "Hinweise: Blickwinkel/Saisonwechsel können Fehlalarme erzeugen; der Vorher/Nachher-Vergleich reduziert diese Effekte."
        ]
        fig = fig_bullets("Wie bewertet das Modell Zerstörung?", bullets, "")
        stamp_page(fig, page); page += 1
        pdf.savefig(fig, dpi=DPI); plt.close(fig)
        if Path(after_img_path).exists():
            fig = fig_full_image_with_caption(after_img_path, "After-Mosaik", caption_after)
            stamp_page(fig, page); page += 1
            pdf.savefig(fig, dpi=DPI); plt.close(fig)
        if Path(heat_img_path).exists():
            fig = fig_full_image_with_caption(heat_img_path, "Heatmap (eingefärbt)", caption_heat)
            stamp_page(fig, page); page += 1
            pdf.savefig(fig, dpi=DPI); plt.close(fig)
        if Path(colorbar_path).exists():
            fig = fig_full_image_with_caption(colorbar_path, "Farbskala", caption_cbar)
            stamp_page(fig, page); page += 1
            pdf.savefig(fig, dpi=DPI); plt.close(fig)
        pages = fig_top_gallery_pages_with_before_after(
            df_all,
            k=min(topk, len(df_all)),
            title="Top 10 Patches (gesamt)",
            caption=caption_top,
            force_last_keys=FORCE_LAST_KEYS
        )
        if pages:
            pg = pages[0]
            stamp_page(pg, page); page += 1
            pdf.savefig(pg, dpi=DPI); plt.close(pg)
    print(f" OK PDF gespeichert: {pdf_path}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--hm_dir", default=HM_DIR)
    ap.add_argument("--out", default=PDF_PATH)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    run(pdf_path=args.out, csv_path=args.csv, hm_dir=args.hm_dir, topk=args.topk)

if __name__ == "__main__":
    main()
