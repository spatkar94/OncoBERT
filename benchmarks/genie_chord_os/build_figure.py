"""Compose the principal three-representation benchmarks into one figure.

Panels A--D use the final mutation-state, standardized sample-level GENIE
retrieval, and embedding-variance figures. Panels E--F use the final MSK-CHORD
OS figure, including exact CHORD TMB as a clinical baseline.
"""
from __future__ import annotations

import json
from pathlib import Path
import os

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


REVISION = Path(os.environ.get("ONCOBERT_WORK_ROOT", Path(__file__).resolve().parents[2]))
OUTPUT = REVISION / "artifacts/three_encoder_composite_benchmark"
SOURCES = {
    "A": REVISION / "artifacts/three_encoder_composite_benchmark/source_panels_large_type/panel_a_mutation_state_large_type.png",
    "B": REVISION / "artifacts/three_encoder_composite_benchmark/source_panels_large_type/panel_b_broad_large_type.png",
    "C": REVISION / "artifacts/three_encoder_composite_benchmark/source_panels_large_type/panel_c_detailed_large_type.png",
    "D": REVISION / "artifacts/three_encoder_composite_benchmark/source_panels_large_type/panel_d_variance_explained_large_type.png",
    "OS": REVISION / "artifacts/slide02_chord_os/plots/chord_os_four_model_benchmark.png",
}

# Pixel rows below the substantive axes in each standalone source. Everything
# below these cutoffs is explanatory fine print that is omitted in the compact
# composite; axis labels and sample counts remain above the cutoffs.
CAPTION_CUTOFFS = {
    "A": 99_999,
    "B": 99_999,
    "C": 99_999,
    "D": 99_999,
}

CANVAS = (6000, 4550)
MARGIN = 55
GAP = 42
LEFT_WIDTH = 2050
RIGHT_X = MARGIN + LEFT_WIDTH + GAP
RIGHT_WIDTH = CANVAS[0] - RIGHT_X - MARGIN


def load_bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Use a common bold font when available, with a portable fallback."""
    candidates = (
        "/usr/local/share/fonts/msttcorefonts/verdanab.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def trim(image: Image.Image, padding: int = 12) -> Image.Image:
    """Trim only uniform white exterior margins, retaining captions and axes."""
    rgb = image.convert("RGB")
    difference = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white"))
    box = difference.getbbox()
    if box is None:
        return rgb
    left, top, right, bottom = box
    return rgb.crop((max(0, left-padding), max(0, top-padding),
                     min(rgb.width, right+padding), min(rgb.height, bottom+padding)))


def paste_contained(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, width, height = box
    fitted = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
    px = x + (width - fitted.width) // 2
    py = y + (height - fitted.height) // 2
    canvas.paste(fitted, (px, py))
    return px, py, fitted.width, fitted.height


def paste_vertically_expanded(
    canvas: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
    vertical_scale: float,
) -> tuple[int, int, int, int]:
    """Fit to width, then modestly elongate vertically within the panel slot."""
    x, y, width, height = box
    fitted = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
    expanded_height = min(height, int(round(fitted.height * vertical_scale)))
    expanded = fitted.resize(
        (fitted.width, expanded_height), Image.Resampling.LANCZOS,
    )
    px = x + (width - expanded.width) // 2
    py = y + (height - expanded.height) // 2
    canvas.paste(expanded, (px, py))
    return px, py, expanded.width, expanded.height


def draw_black_dashed_line(
    draw: ImageDraw.ImageDraw,
    x_left: int,
    x_right: int,
    y: int,
    width: int = 5,
    dash: int = 28,
    gap: int = 16,
) -> None:
    """Draw a high-contrast black dashed reference segment."""
    for start in range(x_left, x_right, dash + gap):
        draw.line(
            (start, y, min(start + dash, x_right), y),
            fill="black", width=width,
        )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    images = {}
    for key in ("A", "B", "C", "D"):
        path = SOURCES[key]
        source = Image.open(path).convert("RGB")
        if CAPTION_CUTOFFS[key] < source.height:
            ImageDraw.Draw(source).rectangle(
                (0, CAPTION_CUTOFFS[key], source.width, source.height), fill="white"
            )
        images[key] = trim(source)
    os_source = Image.open(SOURCES["OS"]).convert("RGB")
    # Split the validated CHORD figure into its C-index and time-dependent
    # AUROC regions so both can expand independently in the taller composite.
    os_top = os_source.crop((0, 0, os_source.width, 1035))
    ImageDraw.Draw(os_top).rectangle((145, 120, 187, 168), fill="white")
    images["E"] = trim(os_top)
    images["F"] = trim(os_source.crop((0, 1020, os_source.width, 2640)))
    canvas = Image.new("RGB", CANVAS, "white")

    left_height = CANVAS[1] - 2*MARGIN
    slot_height = (left_height - 3*GAP) // 4
    placements = {}
    for index, key in enumerate(("A", "B", "C", "D")):
        y = MARGIN + index*(slot_height+GAP)
        placements[key] = paste_contained(canvas, images[key], (MARGIN, y, LEFT_WIDTH, slot_height))
    right_height = CANVAS[1] - 2*MARGIN
    top_height = 1510
    placements["E"] = paste_vertically_expanded(
        canvas, images["E"], (RIGHT_X, MARGIN, RIGHT_WIDTH, top_height),
        vertical_scale=1.12,
    )
    placements["F"] = paste_vertically_expanded(
        canvas, images["F"],
        (RIGHT_X, MARGIN + top_height + GAP, RIGHT_WIDTH,
         right_height - top_height - GAP),
        vertical_scale=1.16,
    )

    # Composite-level panel letters sit in the exterior margin so they remain
    # distinct from the CHORD source figure's own internal subplot lettering.
    draw = ImageDraw.Draw(canvas)
    label_font = load_bold_font(66)
    for key, (x, y, _, _) in placements.items():
        draw.text((max(8, x - 78), y + 4), key, font=label_font, fill="black")

    png = OUTPUT / "three_encoder_retrieval_and_chord_os_composite.png"
    pdf = OUTPUT / "three_encoder_retrieval_and_chord_os_composite.pdf"
    tiff = OUTPUT / "three_encoder_retrieval_and_chord_os_composite.tiff"
    canvas.save(png, dpi=(300, 300), optimize=True)
    canvas.save(pdf, resolution=300.0)
    canvas.save(tiff, dpi=(300, 300), compression="tiff_lzw")
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "complete": True,
        "canvas_pixels": CANVAS,
        "dpi": 300,
        "panels": {
            "A": {
                "analysis": "Mutation-state retrieval by mutation-frequency bin",
                "source": str(SOURCES["A"]),
                "random_guess_auprc_by_frequency_bin": {
                    ">10%": 0.190713,
                    "5-10%": 0.062992,
                    "<5%": 0.019412
                },
                "random_guess_definition": "mean gene-specific positive-class prevalence within each frequency bin",
            },
            "B": {"analysis": "GENIE Broad Cancer Type Retrieval", "source": str(SOURCES["B"])},
            "C": {"analysis": "GENIE Detailed Cancer Type Retrieval", "source": str(SOURCES["C"])},
            "D": {"analysis": "GENIE percent embedding variance explained", "source": str(SOURCES["D"])},
            "E": {"analysis": "MSK-CHORD overall-survival Harrell C-index benchmark", "source": str(SOURCES["OS"])},
            "F": {"analysis": "MSK-CHORD time-dependent AUROC benchmark", "source": str(SOURCES["OS"])},
        },
        "outputs": [str(png), str(pdf), str(tiff)],
    }, indent=2) + "\n")
    print(png)


if __name__ == "__main__":
    main()
