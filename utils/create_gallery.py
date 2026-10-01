"""
score_and_gallery.py
--------------------
Scores generated images against their filename-encoded labels using
BLIP (caption generation) + CLIP ViT-L/14 (semantic similarity),
then renders a ranked HTML gallery.

Filename convention expected:  <idx>_<label1>_<label2>_...jpg
Example:                        0_dog_barking_car_engine_idling.jpg

Usage:
    python score_and_gallery.py \
        --gen_dir  /path/to/generated_images \
        --out_html /path/to/output/gallery.html \
        [--top_k 50]          # only show top-K images (default: all)
        [--num_captions 3]    # BLIP captions per image  (default: 3)
        [--batch_size 8]      # images processed at once (default: 8)
"""

import argparse
import base64
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import clip
import numpy as np
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

# ── Known multi-word labels (longest-match, order matters) ────────────────────
KNOWN_LABELS = [
    "airplane flyby", "ambulance siren", "baby crying", "baby laughter",
    "car engine idling", "cat purring", "chainsawing trees",
    "church bell ringing", "train whistling", "train horning", "cow lowing",
    "dog barking", "elk bugling", "fire truck siren", "gibbon howling",
    "hail", "lawn mowing", "volcano explosion", "stream burbling",
    "waterfall burbling", "orchestra", "owl hooting", "people cheering",
    "people crowd", "people marching", "playing bassoon", "playing drum kit",
    "playing harp", "playing harpsichord", "playing snare drum",
    "playing timbales", "printer printing", "scuba diving", "sea waves",
    "sheep bleating", "singing choir", "skiing", "slot machine",
    "snake hissing", "tractor digging", "underwater bubbling",
    "railroad car, train wagon", "train wheels squealing",
    "baltimore oriole calling", "car engine knocking", "driving snowmobile",
    "wood thrush calling", "hedge trimmer running",
    "ice cream truck, ice cream van", "woodpecker pecking tree",
]

# Aliases: underscore-filename form → canonical label
# Needed for labels that contain commas (can't be encoded as underscores)
LABEL_ALIASES = {
    "ice cream truck ice cream van": "ice cream truck, ice cream van",
    "railroad car train wagon":      "railroad car, train wagon",
}

# Sort longest-first so greedy matching prefers multi-word labels
KNOWN_LABELS_SORTED = sorted(
    KNOWN_LABELS + list(LABEL_ALIASES.keys()),
    key=lambda x: -len(x)
)


# ── Filename → label list ─────────────────────────────────────────────────────

def parse_labels_from_filename(fname: str) -> List[str]:
    """
    Strip extension, drop leading index, reconstruct the label string,
    then greedily match known multi-word labels.

    '0_dog_barking_car_engine_idling.jpg'
    →  ['dog barking', 'car engine idling']
    """
    stem = Path(fname).stem                      # '0_dog_barking_car_engine_idling'
    parts = stem.split("_")

    # Drop leading numeric index if present
    if parts and parts[0].isdigit():
        parts = parts[1:]

    label_str = " ".join(parts)                  # 'dog barking car engine idling'

    found: List[str] = []
    remaining = label_str

    while remaining.strip():
        matched = False
        for label in KNOWN_LABELS_SORTED:
            if remaining.startswith(label):
                found.append(label)
                remaining = remaining[len(label):].strip()
                matched = True
                break
        if not matched:
            # Skip one word if nothing matches (handles unknown tokens)
            tokens = remaining.split(" ", 1)
            remaining = tokens[1] if len(tokens) > 1 else ""

    # Resolve aliases (e.g. comma-containing labels)
    found = [LABEL_ALIASES.get(l, l) for l in found]
    return found if found else parts  # fallback: raw parts


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(device: torch.device):
    print("Loading CLIP ViT-L/14 …")
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    clip_model.eval()

    print("Loading BLIP …")
    blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device).eval()

    return clip_model, clip_preprocess, blip_model, blip_proc


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_image(
    pil_img: Image.Image,
    labels: List[str],
    clip_model,
    clip_preprocess,
    blip_model,
    blip_proc,
    device: torch.device,
    num_captions: int = 3,
) -> Dict:
    """
    Returns a dict with:
      captions       – list of BLIP captions
      per_label_sim  – best caption↔label cosine sim for each label
      gm_score       – geometric mean of per_label similarities  (0-1, higher=better)
      top_label      – label with highest similarity
    """
    # ── BLIP captions ──────────────────────────────────────────────────────────
    inputs = blip_proc(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = blip_model.generate(
            **inputs,
            max_length=40,
            num_beams=max(num_captions, 3),
            num_return_sequences=num_captions,
            early_stopping=True,
        )
    captions = blip_proc.batch_decode(ids, skip_special_tokens=True)

    # ── CLIP encode ────────────────────────────────────────────────────────────
    cap_tokens = clip.tokenize(captions, truncate=True).to(device)
    lbl_prompts = [f"a photo of {l}" for l in labels]
    lbl_tokens = clip.tokenize(lbl_prompts, truncate=True).to(device)

    with torch.no_grad():
        cap_emb = clip_model.encode_text(cap_tokens).float()
        lbl_emb = clip_model.encode_text(lbl_tokens).float()

    cap_emb = cap_emb / cap_emb.norm(dim=-1, keepdim=True)
    lbl_emb = lbl_emb / lbl_emb.norm(dim=-1, keepdim=True)

    sims = (cap_emb @ lbl_emb.T)          # (num_captions, num_labels)
    best_per_label = sims.max(dim=0).values.cpu()   # (num_labels,)

    scores = best_per_label.clamp(min=1e-6)
    gm = float(torch.exp(torch.mean(torch.log(scores))))

    top_idx = int(best_per_label.argmax())

    return {
        "captions": captions,
        "per_label_sim": best_per_label.tolist(),
        "gm_score": gm,
        "top_label": labels[top_idx],
    }


# ── Image → base64 for inline HTML ───────────────────────────────────────────

def img_to_b64(pil_img: Image.Image, max_side: int = 400) -> str:
    img = pil_img.copy()
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ── HTML rendering ────────────────────────────────────────────────────────────

def render_html(results: List[Dict], out_path: str):
    def score_bar(v: float) -> str:
        pct = round(v * 100, 1)
        hue = int(v * 120)   # red→green
        return (
            f'<div class="bar-wrap"><div class="bar" style="width:{pct}%;'
            f'background:hsl({hue},70%,45%)"></div>'
            f'<span class="bar-val">{pct}%</span></div>'
        )

    cards_html = ""
    for rank, r in enumerate(results, 1):
        labels_html = ""
        for lbl, sim in zip(r["labels"], r["per_label_sim"]):
            star = " ★" if lbl == r["top_label"] else ""
            labels_html += (
                f'<div class="label-row">'
                f'<span class="lbl-name">{lbl}{star}</span>'
                f'{score_bar(sim)}</div>'
            )

        captions_html = "".join(
            f'<li class="cap">"{c}"</li>' for c in r["captions"]
        )

        gm_pct = round(r["gm_score"] * 100, 1)
        gm_hue = int(r["gm_score"] * 120)

        cards_html += f"""
        <article class="card" data-score="{r['gm_score']:.4f}">
            <div class="rank">#{rank}</div>
            <div class="img-wrap">
                <img src="data:image/jpeg;base64,{r['b64']}" alt="{r['filename']}" loading="lazy"/>
                <div class="gm-badge" style="background:hsl({gm_hue},65%,42%)">
                    {gm_pct}%
                </div>
            </div>
            <div class="meta">
                <p class="fname" title="{r['filename']}">{r['filename']}</p>
                <div class="section-title">Label Similarities</div>
                {labels_html}
                <div class="section-title" style="margin-top:10px">BLIP Captions</div>
                <ul class="caps-list">{captions_html}</ul>
            </div>
        </article>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Generated Image Gallery — Ranked by VLM Score</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  :root {{
    --bg:        #0d0f14;
    --surface:   #161a22;
    --border:    #252b38;
    --accent:    #e8ff5a;
    --text:      #e2e6f0;
    --muted:     #6b7690;
    --radius:    12px;
  }}

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    min-height: 100vh;
  }}

  /* ── Header ── */
  header {{
    padding: 48px 40px 28px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: flex-end;
    gap: 24px;
    flex-wrap: wrap;
  }}
  header h1 {{
    font-size: clamp(1.6rem, 4vw, 2.8rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
  }}
  header h1 span {{ color: var(--accent); }}
  .meta-bar {{
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    margin-left: auto;
  }}

  /* ── Controls ── */
  .controls {{
    padding: 20px 40px;
    display: flex;
    gap: 16px;
    align-items: center;
    flex-wrap: wrap;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }}
  .controls label {{ font-size: 0.82rem; color: var(--muted); }}
  .controls input[type=range] {{
    accent-color: var(--accent);
    cursor: pointer;
  }}
  #threshold-val {{
    font-family: 'DM Mono', monospace;
    color: var(--accent);
    font-size: 0.82rem;
    min-width: 36px;
  }}
  #count-display {{
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    margin-left: auto;
  }}

  /* ── Grid ── */
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 24px;
    padding: 32px 40px;
  }}

  /* ── Card ── */
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: transform .2s ease, border-color .2s ease;
    position: relative;
  }}
  .card:hover {{
    transform: translateY(-4px);
    border-color: var(--accent);
  }}
  .card.hidden {{ display: none; }}

  .rank {{
    position: absolute;
    top: 10px; left: 10px;
    background: rgba(0,0,0,.7);
    color: var(--accent);
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 2px 7px;
    border-radius: 4px;
    z-index: 2;
  }}

  .img-wrap {{
    position: relative;
    background: #0a0c10;
    aspect-ratio: 1 / 1;
    overflow: hidden;
  }}
  .img-wrap img {{
    width: 100%; height: 100%;
    object-fit: contain;
    display: block;
  }}
  .gm-badge {{
    position: absolute;
    bottom: 10px; right: 10px;
    color: #fff;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 3px 9px;
    border-radius: 6px;
  }}

  .meta {{
    padding: 16px;
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .fname {{
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .section-title {{
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--muted);
    font-weight: 700;
    margin-top: 4px;
  }}

  .label-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.78rem;
    margin-top: 4px;
  }}
  .lbl-name {{
    min-width: 150px;
    max-width: 150px;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .bar-wrap {{
    flex: 1;
    display: flex;
    align-items: center;
    gap: 6px;
    height: 6px;
    background: #1e2530;
    border-radius: 3px;
    overflow: visible;
    position: relative;
  }}
  .bar {{
    height: 100%;
    border-radius: 3px;
    transition: width .4s ease;
  }}
  .bar-val {{
    position: absolute;
    right: -38px;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: var(--muted);
    white-space: nowrap;
  }}

  .caps-list {{
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 4px;
  }}
  .cap {{
    font-family: 'DM Mono', monospace;
    font-size: 0.67rem;
    color: var(--muted);
    line-height: 1.45;
    border-left: 2px solid var(--border);
    padding-left: 8px;
  }}

  /* ── Footer ── */
  footer {{
    text-align: center;
    padding: 32px;
    color: var(--muted);
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<header>
  <div>
    <h1>Image Gallery <span>/ VLM Ranked</span></h1>
    <p style="color:var(--muted);font-size:.8rem;margin-top:6px;font-family:'DM Mono',monospace">
      Scored via BLIP captions × CLIP ViT-L/14 · ranked by geometric mean similarity
    </p>
  </div>
  <div class="meta-bar">{len(results)} images evaluated</div>
</header>

<div class="controls">
  <label for="threshold-slider">Min score filter:</label>
  <input type="range" id="threshold-slider" min="0" max="100" value="0" step="1"/>
  <span id="threshold-val">0%</span>
  <span id="count-display">{len(results)} shown</span>
</div>

<div class="grid" id="gallery">
{cards_html}
</div>

<footer>
  Generated by score_and_gallery.py &nbsp;·&nbsp; BLIP + CLIP ViT-L/14
</footer>

<script>
  const slider = document.getElementById('threshold-slider');
  const thVal  = document.getElementById('threshold-val');
  const countD = document.getElementById('count-display');
  const cards  = Array.from(document.querySelectorAll('.card'));

  slider.addEventListener('input', () => {{
    const t = +slider.value / 100;
    thVal.textContent = slider.value + '%';
    let visible = 0;
    cards.forEach(c => {{
      const show = +c.dataset.score >= t;
      c.classList.toggle('hidden', !show);
      if (show) visible++;
    }});
    countD.textContent = visible + ' shown';
  }});
</script>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Gallery saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir",      required=True, help="Folder of generated images")
    parser.add_argument("--out_html",     default="gallery.html")
    parser.add_argument("--top_k",        type=int, default=None, help="Show only top-K images")
    parser.add_argument("--num_captions", type=int, default=3)
    parser.add_argument("--batch_size",   type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Gather image paths ─────────────────────────────────────────────────────
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    img_paths = sorted([
        p for p in Path(args.gen_dir).iterdir()
        if p.suffix.lower() in exts
    ])
    print(f"Found {len(img_paths)} images in {args.gen_dir}")

    clip_model, clip_preprocess, blip_model, blip_proc = load_models(device)

    results = []
    for i, path in enumerate(img_paths):
        print(f"[{i+1}/{len(img_paths)}] {path.name}")
        labels = parse_labels_from_filename(path.name)
        if not labels:
            print(f"  ⚠ Could not parse labels, skipping.")
            continue

        pil_img = Image.open(path).convert("RGB")
        score_dict = score_image(
            pil_img, labels,
            clip_model, clip_preprocess,
            blip_model, blip_proc,
            device, num_captions=args.num_captions,
        )

        results.append({
            "filename":     path.name,
            "labels":       labels,
            "captions":     score_dict["captions"],
            "per_label_sim":score_dict["per_label_sim"],
            "gm_score":     score_dict["gm_score"],
            "top_label":    score_dict["top_label"],
            "b64":          img_to_b64(pil_img),
        })

    # ── Sort best first ────────────────────────────────────────────────────────
    results.sort(key=lambda x: x["gm_score"], reverse=True)

    if args.top_k:
        results = results[: args.top_k]

    render_html(results, args.out_html)


if __name__ == "__main__":
    main()