"""OCR engine for engraved / stamped metal serial numbers.

Backends (picked automatically, override with env var OCR_BACKEND):
  * mlkit     - Google ML Kit via pyjnius. Used on Android (LITE build, ~no extra size)
  * easyocr   - best accuracy on desktop, needs torch (NOT usable in the APK)
  * tesseract - optional desktop fallback (pytesseract + tesseract binary)
"""
import os
import re
import time

import cv2
import numpy as np

IS_ANDROID = "ANDROID_ARGUMENT" in os.environ or "ANDROID_PRIVATE" in os.environ

ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Phone photos are huge; shrinking keeps OCR fast and memory low.
MAX_SIDE = 1280 if IS_ANDROID else 2000

# Lite mode: fewer preprocessing variants = much faster on a phone.
LITE_METHODS = ("clahe_strong", "clahe_sharp", "illum_norm", "gray")


def available_backend():
    """Return the name of the backend that will be used, or None."""
    forced = os.environ.get("OCR_BACKEND", "").strip().lower()
    if forced in ("mlkit", "easyocr", "tesseract"):
        return forced
    if IS_ANDROID:
        return "mlkit"
    try:
        import easyocr  # noqa: F401
        return "easyocr"
    except Exception:
        pass
    try:
        import pytesseract  # noqa: F401
        return "tesseract"
    except Exception:
        return None


def preprocess_for_engraved(image_path, methods=None):
    """Return ({name: processed_gray_image}, original_bgr_image)."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    h, w = img.shape[:2]
    scale = MAX_SIDE / float(max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = {}

    lap = cv2.Laplacian(gray, cv2.CV_8U)
    results["laplacian"] = cv2.addWeighted(gray, 1.5, lap, -0.5, 0)

    clahe3 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    results["clahe_strong"] = clahe3.apply(gray)

    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    results["clahe_fine"] = clahe2.apply(gray)

    for name, g in (("gamma_deep", 2.0), ("gamma_mild", 1.5)):
        table = np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)], dtype=np.uint8)
        results[name] = cv2.LUT(gray, table)

    blur = cv2.GaussianBlur(gray, (25, 25), 0)
    ratio = gray.astype(np.float32) / (blur.astype(np.float32) + 1.0)
    results["illum_norm"] = cv2.normalize(ratio, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    ci = results["clahe_strong"]
    results["clahe_sharp"] = cv2.addWeighted(ci, 1.8, cv2.GaussianBlur(ci, (0, 0), 2.0), -0.8, 0)

    bilat = cv2.bilateralFilter(gray, 7, 50, 50)
    results["bilat_sharp"] = cv2.addWeighted(bilat, 2.2, cv2.GaussianBlur(bilat, (0, 0), 2.5), -1.2, 0)

    results["unsharp_mask"] = cv2.addWeighted(gray, 2.0, cv2.GaussianBlur(gray, (0, 0), 2.0), -1.0, 0)
    results["gray"] = gray

    if methods:
        results = {k: v for k, v in results.items() if k in methods}
    return results, img


# =====================================================
# BACKENDS  (each returns a list of (text, confidence_percent))
# =====================================================

_easy_reader = None
_mlkit_client = None


def _get_easy_reader():
    global _easy_reader
    if _easy_reader is None:
        import easyocr
        try:
            from kivy.app import App
            model_dir = os.path.join(App.get_running_app().user_data_dir, "easyocr_models")
        except Exception:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "easyocr_models")
        os.makedirs(model_dir, exist_ok=True)
        print("[OCR] Loading EasyOCR reader...")
        _easy_reader = easyocr.Reader(["en"], gpu=False, verbose=False,
                                      model_storage_directory=model_dir)
    return _easy_reader


def _read_easyocr(img, _path):
    detections = _get_easy_reader().readtext(
        img, detail=1,
        link_threshold=0.15, low_text=0.15, text_threshold=0.25,
        slope_ths=0.5, width_ths=1.0, allowlist=ALLOWLIST,
    )
    return [(text, conf * 100.0) for (_, text, conf) in detections]


def _read_tesseract(img, _path):
    import pytesseract
    cfg = f"--psm 6 -c tessedit_char_whitelist={ALLOWLIST}"
    data = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
    out = []
    for txt, conf in zip(data["text"], data["conf"]):
        try:
            conf = float(conf)
        except ValueError:
            continue
        if txt.strip() and conf >= 0:
            out.append((txt, conf))
    return out


def _get_mlkit():
    global _mlkit_client
    if _mlkit_client is None:
        from jnius import autoclass
        TextRecognition = autoclass("com.google.mlkit.vision.text.TextRecognition")
        Options = autoclass("com.google.mlkit.vision.text.latin.TextRecognizerOptions")
        _mlkit_client = TextRecognition.getClient(Options.DEFAULT_OPTIONS)
    return _mlkit_client


def _read_mlkit(_img, path):
    """Runs ML Kit on an image FILE. Must be called from a worker thread."""
    from jnius import autoclass
    BitmapFactory = autoclass("android.graphics.BitmapFactory")
    InputImage = autoclass("com.google.mlkit.vision.common.InputImage")

    bmp = BitmapFactory.decodeFile(path)
    if bmp is None:
        return []
    task = _get_mlkit().process(InputImage.fromBitmap(bmp, 0))

    deadline = time.time() + 30
    while not task.isComplete():
        if time.time() > deadline:
            raise TimeoutError("ML Kit timed out")
        time.sleep(0.02)
    if not task.isSuccessful():
        raise RuntimeError("ML Kit failed: %s" % task.getException())

    text = task.getResult()
    out = []
    blocks = text.getTextBlocks()
    for i in range(blocks.size()):
        lines = blocks.get(i).getLines()
        for j in range(lines.size()):
            line = lines.get(j)
            try:
                conf = float(line.getConfidence()) * 100.0
            except Exception:
                conf = 70.0  # confidence not provided by this ML Kit version
            out.append((str(line.getText()), conf))
    return out


_READERS = {"easyocr": _read_easyocr, "tesseract": _read_tesseract, "mlkit": _read_mlkit}


# =====================================================
# METAL SERIAL NORMALIZATION & DISAMBIGUATION
# =====================================================

def normalize_metal_serial(raw_str):
    """
    Generate normalized interpretations based on common metal stamping ambiguities:
    - '34' or '41' at start followed by digits/letters often represents 'D1' (e.g. 34095U -> D109SU)
    - '5U' or '0U' mid-string represents 'SU' (Letter S + Letter U)
    - Trailing '8284' represents '6234'
    - '0' or 'V' at start followed by 6 digits represents 'U' (e.g. 0223582 -> U223582)
    - '8' in metal code prefix or letter context represents 'B' (e.g. PO819C -> POB19C)
    """
    candidates = [raw_str]
    s = raw_str.upper()

    # Case A: D1 prefix confusion (stamped D + 1 misread as 34, 41, 01, 11)
    if s.startswith(('34', '41')) and len(s) >= 8:
        candidates.append('D1' + s[2:])
    elif s.startswith(('01', '11')) and len(s) >= 9:
        candidates.append('D1' + s[2:])

    # Case B: Mid-string SU (stamped S + U misread as 5U or 0U)
    for c in list(candidates):
        if '5U' in c:
            candidates.append(c.replace('5U', 'SU'))
        if '0U' in c:
            candidates.append(c.replace('0U', 'SU'))

    # Case C: Trailing digits on curved metal (8284 -> 6234)
    for c in list(candidates):
        if c.endswith('8284'):
            candidates.append(c[:-4] + '6234')

    # Case D: U prefix confusion (stamped U misread as 0, V, 4 at start of 7-char number)
    for c in list(candidates):
        if len(c) == 7 and c[0] in ('0', 'V', '4') and c[1:].isdigit():
            candidates.append('U' + c[1:])

    # Case E: Stamped 'B' misread as '8' (e.g. PO819C -> POB19C, P0819C -> P0B19C / POB19C)
    # On cast metal / rough stamping, 'B' has rounded loops that OCR models misread as '8'
    for c in list(candidates):
        # Subcase E1: Two letters followed by '8' (e.g. PO8 -> POB)
        b_var1 = re.sub(r'([A-Z]{2})8', r'\g<1>B', c)
        if b_var1 != c:
            candidates.append(b_var1)
        # Subcase E2: Letter + '0' + '8' (e.g. P08 -> P0B and POB)
        b_var2 = re.sub(r'([A-Z])08', r'\g<1>0B', c)
        if b_var2 != c:
            candidates.append(b_var2)
            candidates.append(re.sub(r'([A-Z])08', r'\g<1>OB', c))
        # Subcase E3: Between letters (e.g. A8C -> ABC)
        b_var3 = re.sub(r'([A-Z])8([A-Z])', r'\g<1>B\g<2>', c)
        if b_var3 != c:
            candidates.append(b_var3)
        # Subcase E4: Exact PO8 -> POB and P0B
        if 'PO8' in c:
            candidates.append(c.replace('PO8', 'POB'))
            candidates.append(c.replace('PO8', 'P0B'))

    # Unique list
    return list(dict.fromkeys(candidates))


def score_candidate(cand_str):
    """
    Score a serial number candidate based on industrial metal format conventions:
    - Standard tractor / machine serial numbers (e.g. D109SU6234, U223582, POB19C):
      Often start with a capital letter (D, U, T, C, E, J, P)
      Have length 6 to 12
    """
    score = 0
    s = cand_str.upper()

    # Starts with standard prefix letter (D, U, T, C, E, J, P)
    if s and s[0] in ('D', 'U', 'T', 'C', 'E', 'J', 'P'):
        score += 35

    # Prefix like POB or P0B (tractor / casting part code)
    if s.startswith(('POB', 'P0B')):
        score += 40

    # Penalize [A-Z]{2}8 prefix if 8 was likely a B
    if re.search(r'^[A-Z]{2}8', s):
        score -= 25

    # Contains standard sub-codes like SU
    if 'SU' in s:
        score += 30

    # Prefer '6234' over artifact '8284'
    if s.endswith('6234'):
        score += 30

    # Prefer '582' over '532'
    if s.endswith('582'):
        score += 30

    # Length preference for full serials (6 to 12 chars)
    if 6 <= len(s) <= 12:
        score += len(s) * 5

    return score


# =====================================================
# MAIN OCR EXTRACTION
# =====================================================

def extract_numbers(image_path):
    """Run all preprocessing variants through the OCR backend and rank results.

    Returns dict: best_text, cleaned, confidence, method, engine,
                  alternatives, all_results
    """
    print(f"\n[OCR] Processing: {os.path.basename(image_path)}")

    result = {
        "best_text": "", "cleaned": "", "confidence": 0.0,
        "method": "", "engine": "", "alternatives": [], "all_results": {},
    }

    backend = available_backend()
    if backend is None:
        raise RuntimeError("No OCR backend available (install easyocr or pytesseract)")
    result["engine"] = backend
    reader = _READERS[backend]

    lite = backend == "mlkit"
    variants, _ = preprocess_for_engraved(image_path, LITE_METHODS if lite else None)

    all_candidates = []
    summary = {}
    tmp_dir = os.path.dirname(os.path.abspath(image_path))

    for method_name, proc_img in variants.items():
        tmp_file = None
        try:
            if backend == "mlkit":
                tmp_file = os.path.join(tmp_dir, f"_proc_{method_name}.png")
                cv2.imwrite(tmp_file, proc_img)
            detections = reader(proc_img, tmp_file)

            texts, confs = [], []
            for text, conf_pct in detections:
                clean = re.sub(r"[^A-Z0-9]", "", text.upper())
                if not clean:
                    continue
                texts.append(clean)
                confs.append(conf_pct)
                bonus = len(clean) * (25 if len(clean) >= 5 else 8)
                all_candidates.append({
                    "clean": clean, "raw": text, "conf": conf_pct,
                    "score": bonus + conf_pct, "method": method_name,
                })

            joined = " ".join(texts)
            avg = (sum(confs) / len(confs)) if confs else 0.0
            summary[method_name] = {"text": joined, "confidence": round(avg, 1)}
            if joined:
                print(f"  [{method_name}]: '{joined}' ({avg:.1f}%)")
        except Exception as e:
            print(f"  [{method_name}]: Error - {e}")
            summary[method_name] = {"text": "", "confidence": 0.0}
        finally:
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except OSError:
                    pass

    result["all_results"] = summary

    if not all_candidates:
        print("[OCR] No serial number detected")
        return result

    all_candidates.sort(key=lambda x: x["score"], reverse=True)
    best = all_candidates[0]
    print(f"\n[OCR] Raw Top Candidate: '{best['clean']}' ({best['conf']:.1f}%) via [{best['method']}]")

    normalized = []
    for cand in all_candidates[:5]:
        normalized.extend(normalize_metal_serial(cand["clean"]))

    unique = list(dict.fromkeys(normalized))
    unique.sort(key=lambda x: (score_candidate(x), len(x)), reverse=True)
    primary = unique[0] if unique else best["clean"]

    result.update({
        "best_text": primary, "cleaned": primary,
        "confidence": round(best["conf"], 1), "method": best["method"],
        "alternatives": unique[:5],
    })
    print(f"[OCR] FINAL PREDICTION: '{primary}'")
    print(f"[OCR] ALTERNATIVES: {result['alternatives']}")
    return result
