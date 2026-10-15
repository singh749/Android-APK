"""Local smoke test for the parts of ocr_engine that do not need Android.

Run with:  /tmp/venv/bin/python tests/test_engine_offline.py
(kept in the repo so CI can run it too)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw  # noqa: E402

import ocr_engine  # noqa: E402


def make_test_image(path):
    """A 'metal plate' with an engraved looking serial number."""
    img = Image.new("RGB", (900, 500), (150, 152, 156))
    d = ImageDraw.Draw(img)
    for y in range(500):          # brushed metal stripes
        shade = 140 + (y % 7) * 3
        d.line([(0, y), (900, y)], fill=(shade, shade + 1, shade + 3))
    d.text((60, 220), "D109SU6234", fill=(60, 62, 66))       # engraved (dark)
    d.text((60, 300), "POB19C", fill=(205, 207, 210))        # raised (light)
    img.save(path)
    return path


def main():
    tmp = tempfile.mkdtemp(prefix="ocr_test_")
    image = make_test_image(os.path.join(tmp, "plate.jpg"))
    print("test image:", image)

    # 0) backend probing must not explode on a plain PC
    print("autodetected backend:", ocr_engine.available_backend())

    # 1) preprocessing variants (Pillow path used on Android)
    variants = ocr_engine.preprocess_pillow(image)
    print("variants:", sorted(variants))
    assert "crop_2x" in variants and "inverted" in variants and "binary" in variants
    for name, im in variants.items():
        assert im.size[0] > 0 and im.size[1] > 0, name
    # every variant must be writable as jpeg (that is what the Android reads)
    for name, im in variants.items():
        out = os.path.join(tmp, f"{name}.jpg")
        im.save(out, format="JPEG", quality=92)
        assert os.path.getsize(out) > 0, name
    print("preprocessing OK")

    # 2) ranking pipeline with a stubbed backend (ML Kit is Android only)
    def fake_reader(_img, _path):
        return [("D1095U6234", 78.0), ("POB19C", 64.0), ("", 0.0)]

    ocr_engine._READERS["fake"] = fake_reader
    ocr_engine.available_backend = lambda: "fake"

    result = ocr_engine.extract_numbers(image)
    print("result:", {k: v for k, v in result.items() if k != "all_results"})
    assert result["engine"] == "fake"
    assert result["best_text"], "no primary candidate"
    assert result["confidence"] > 0
    assert isinstance(result["alternatives"], list)

    # normalisation: '5U' is the classic misread of 'SU'
    assert "D109SU6234" in result["alternatives"], result["alternatives"]
    print("ranking OK")

    # 3) the score/normalise helpers themselves
    assert ocr_engine.score_candidate("POB19C") > ocr_engine.score_candidate("PO819C")
    assert "D109SU6234" in ocr_engine.normalize_metal_serial("D1095U6234")
    print("helpers OK")

    # 4) android-only entry points must be import safe on a PC
    assert ocr_engine.HAVE_PIL is True
    assert ocr_engine.available_backend() == "fake"  # still patched above
    print("\nALL OFFLINE CHECKS PASSED")


if __name__ == "__main__":
    main()
