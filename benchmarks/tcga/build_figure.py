#!/usr/bin/env python3
"""Assemble the final four-panel TCGA representation benchmark figure.

This script intentionally reuses the previously generated, validated panels.  It
only performs layout operations: removal of panel-specific footer notes,
whitespace trimming, aspect-ratio-preserving scaling, and panel lettering.
"""

from __future__ import annotations

from pathlib import Path
import os

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


ROOT = Path(os.environ.get("ONCOBERT_WORK_ROOT", Path(__file__).resolve().parents[2]))
PANELS = {
    "A": ROOT / "artifacts/slide03_tcga_umap/tcga_cancer_type_umap_labeled.png",
    "B": ROOT / "artifacts/standardized_sample_level_retrieval/tcga/broad/tcga_broad_three_model_retrieval.png",
    "C": ROOT / "artifacts/standardized_sample_level_retrieval/tcga/detailed/tcga_detailed_three_model_retrieval.png",
    "D": ROOT / "artifacts/standardized_sample_level_retrieval/tcga/icdo3/tcga_icdo3_three_model_retrieval.png",
    "E": ROOT / "artifacts/slide03_04_tcga_retrieval/tcga_molecular_subtype_per_cancer_supplement.png",
}
OUTDIR = ROOT / "artifacts/tcga_composite_benchmark"
REPRO = ROOT / "reproduced_figures"
STEM = "tcga_embedding_and_classification_composite"


def load_bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Use a common bold font when available, with a portable fallback."""
    candidates = (
        "/usr/local/share/fonts/msttcorefonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def white_out_footer(image: Image.Image, fraction: float | None) -> Image.Image:
    """Remove old explanatory footers while retaining axes and legends."""
    image = image.convert("RGB")
    if fraction is None:
        return image
    y = int(image.height * fraction)
    ImageDraw.Draw(image).rectangle((0, y, image.width, image.height), fill="white")
    return image


def trim_white(image: Image.Image, pad: int = 14) -> Image.Image:
    """Crop only near-white exterior whitespace."""
    rgb = image.convert("RGB")
    bg = Image.new("RGB", rgb.size, "white")
    diff = ImageChops.difference(rgb, bg).convert("L")
    # Ignore tiny antialiasing/compression differences in otherwise white areas.
    diff = diff.point(lambda p: 255 if p > 8 else 0)
    box = diff.getbbox()
    if box is None:
        return rgb
    left, top, right, bottom = box
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(rgb.width, right + pad)
    bottom = min(rgb.height, bottom + pad)
    return rgb.crop((left, top, right, bottom))


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Fit inside a slot without stretching."""
    return ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)


def load_panel(label: str) -> Image.Image:
    image = Image.open(PANELS[label])
    # A has a very short footer; B/C/D have longer methods notes; E has no footer.
    footer_fraction = {
        "A": 0.974,
        "B": 0.947,
        "C": 0.947,
        "D": 0.947,
        "E": None,
    }[label]
    return trim_white(white_out_footer(image, footer_fraction))


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    REPRO.mkdir(parents=True, exist_ok=True)

    # Two conventional columns hold the four graphical panels.  The wider
    # right-hand column is reserved for the vertically dense subtype table and
    # spans both rows, so its typography remains readable in the composite.
    canvas_w, canvas_h = 9400, 4600
    margin_x, margin_y = 100, 90
    gap_x, gap_y = 100, 100
    left_w, middle_w, table_w = 2500, 2500, 4000
    cell_h = (canvas_h - 2 * margin_y - gap_y) // 2
    label_band = 82

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_bold_font(68)

    middle_x = margin_x + left_w + gap_x
    table_x = middle_x + middle_w + gap_x
    slots = {
        "A": (margin_x, margin_y, left_w, cell_h),
        "B": (middle_x, margin_y, middle_w, cell_h),
        "C": (margin_x, margin_y + cell_h + gap_y, left_w, cell_h),
        "D": (middle_x, margin_y + cell_h + gap_y, middle_w, cell_h),
        "E": (table_x, margin_y, table_w, 2 * cell_h + gap_y),
    }

    for label, (x, y, slot_w, slot_h) in slots.items():
        panel = fit(load_panel(label), (slot_w, slot_h - label_band))
        px = x + (slot_w - panel.width) // 2
        py = y + label_band + (slot_h - label_band - panel.height) // 2
        canvas.paste(panel, (px, py))
        draw.text((x + 8, y + 2), label, fill="black", font=font)

    outputs = [OUTDIR / f"{STEM}.png", OUTDIR / f"{STEM}.pdf", OUTDIR / f"{STEM}.tiff"]
    canvas.save(outputs[0], dpi=(300, 300), optimize=True)
    canvas.save(outputs[1], "PDF", resolution=300.0)
    canvas.save(outputs[2], "TIFF", dpi=(300, 300), compression="tiff_lzw")

    for src in outputs:
        dst = REPRO / src.name
        dst.write_bytes(src.read_bytes())

    print(f"Wrote composite figure to {OUTDIR}")


if __name__ == "__main__":
    main()
