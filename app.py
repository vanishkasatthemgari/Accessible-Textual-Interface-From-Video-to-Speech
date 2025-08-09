import os
import cv2
import numpy as np
from flask import (
    Flask, render_template, request, redirect,
    url_for, send_from_directory, flash
)
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
from gtts import gTTS
from googletrans import Translator
from collections import Counter
import matplotlib.pyplot as plt

#Config
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"  # change if needed

UPLOAD_FOLDER = "D:\\Text video to speech - Copy\\upload"
STATIC_FOLDER = "D:\\Text video to speech - Copy\\static"
IMAGE_FOLDER  = "D:\\Text video to speech - Copy\\images"

ALLOWED_EXTENSIONS = {"mp4", "jpeg", "jpg", "png"}

LANGUAGE_CHOICES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "bn": "Bengali",
    "fr": "French",
}

# ─────────────────────────────────────────
#  Flask setup
# ─────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "1234"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

for path in (UPLOAD_FOLDER, STATIC_FOLDER, IMAGE_FOLDER):
    os.makedirs(path, exist_ok=True)

# ─────────────────────────────────────────
#  (Optional) super-resolution model
# ─────────────────────────────────────────
sr = cv2.dnn_superres.DnnSuperResImpl_create()
model_path = "D:\Text video to speech - Copy\EDSR_x4.pb"
if os.path.exists(model_path):
    sr.readModel(model_path)
    sr.setModel("edsr", 2)
else:
    print("Super-Resolution model not found – continuing without it.")

# ─────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def frame_capture(video_path: str):
    """Extract every 50-th frame and save to images/ directory."""
    vidcap = cv2.VideoCapture(video_path)
    count, idx, saved_frames = 0, 1, []

    while vidcap.isOpened():
        success, frame = vidcap.read()
        if not success:
            break
        if count % 50 == 0:
            fpath = os.path.join(IMAGE_FOLDER, f"{idx}.jpg")
            cv2.imwrite(fpath, frame)
            saved_frames.append(fpath)
            idx += 1
        count += 1

    vidcap.release()
    print(f"[DEBUG] Extracted {len(saved_frames)} frames from video.")
    return saved_frames


def preprocess_image(img_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.medianBlur(img, 3)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return img


def stitch_images(img_paths):
    imgs = [cv2.imread(p) for p in img_paths if cv2.imread(p) is not None]
    if not imgs:
        return None
    if len(imgs) == 1:
        return imgs[0]
    stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch(imgs)
    return stitched if status == cv2.Stitcher_OK else np.vstack(imgs)


def calculate_metrics(ground_truth: str, ocr_output: str):
    gt_chars  = list(ground_truth)
    ocr_chars = list(ocr_output)

    gt_counter  = Counter(gt_chars)
    ocr_counter = Counter(ocr_chars)

    correct   = sum((gt_counter & ocr_counter).values())
    total_gt  = len(gt_chars)
    total_ocr = len(ocr_chars)

    accuracy  = correct / total_gt  if total_gt  else 0
    precision = correct / total_ocr if total_ocr else 0
    recall    = correct / total_gt  if total_gt  else 0
    return accuracy, precision, recall


def plot_metrics(accuracy, precision, recall, detected_lang_code, folder="static"):
    """Save bar-chart image as static/metrics_<lang>.png (not shown to user)."""
    metrics = [accuracy, precision, recall]
    labels  = ["Accuracy", "Precision", "Recall"]
    colors  = ["#4caf50", "#2196f3", "#ff9800"]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, metrics, color=colors)
    plt.ylim(0, 100)
    plt.title(f"OCR Metrics ({detected_lang_code.upper()})")
    plt.ylabel("Percentage")
    plt.tight_layout()

    fname = f"metrics_{detected_lang_code}.png"
    plt.savefig(os.path.join(folder, fname))
    plt.close()

# ─────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", languages=LANGUAGE_CHOICES)


@app.route("/upload", methods=["POST"])
def upload():
    # 1. validate file ---------------------------------------------------
    file = request.files.get("file")
    if not file or file.filename == "" or not allowed_file(file.filename):
        flash("Please select a valid MP4 / image file.")
        return redirect(url_for("index"))

    out_lang = request.form.get("out_lang", "en").lower()
    if out_lang not in LANGUAGE_CHOICES:
        flash("Selected output language is not supported.")
        return redirect(url_for("index"))

    # 2. save upload -----------------------------------------------------
    filename   = secure_filename(file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(video_path)

    # 3. extract & stitch frames ----------------------------------------
    frames   = frame_capture(video_path)
    stitched = stitch_images(frames)
    if stitched is None:
        flash("Error processing video frames.")
        return redirect(url_for("index"))

    stitched_path = os.path.join(STATIC_FOLDER, "stitched.jpg")
    cv2.imwrite(stitched_path, stitched)

    # 4. pre-process for OCR --------------------------------------------
    processed       = preprocess_image(stitched_path)
    processed_path  = os.path.join(STATIC_FOLDER, "processed.jpg")
    cv2.imwrite(processed_path, processed)

    # 5. OCR -------------------------------------------------------------
    extracted_text = pytesseract.image_to_string(
        Image.open(processed_path).convert("RGB"),
        lang="eng+hin+tel+tam+fra",
        config="--psm 6",
    ).strip()

    with open(os.path.join(STATIC_FOLDER, "ocr.txt"), "w", encoding="utf-8") as f:
        f.write(extracted_text)

    # 6. translate -------------------------------------------------------
    translator      = Translator()
    translated      = translator.translate(extracted_text, dest=out_lang)
    translated_text = translated.text
    detected_lang   = translated.src  # e.g. 'hin', 'eng', ...

    # 7. evaluation metrics (console only) ------------------------------
    gt_path = "D:/Text video to speech - Copy/ground_truth.txt"               # adjust if your file lives elsewhere
    if os.path.exists(gt_path) and extracted_text:
        with open(gt_path, encoding="utf-8") as f:
            gt_text = f.read().strip()

        if gt_text:
            acc, prec, rec = calculate_metrics(gt_text, extracted_text)
            acc  *= 100
            prec *= 100
            rec  *= 100
            print(f"[DEBUG] OCR Accuracy:  {acc:.2f}%")
            print(f"[DEBUG] OCR Precision: {prec:.2f}%")
            print(f"[DEBUG] OCR Recall:    {rec:.2f}%")
            plot_metrics(acc, prec, rec, detected_lang)  # saved to static/, not displayed

    # 8. TTS -------------------------------------------------------------
    audio_filename = f"tts_{out_lang}.mp3"
    gTTS(text=translated_text, lang=out_lang).save(
        os.path.join(STATIC_FOLDER, audio_filename)
    )

    # 9. render result page ---------------------------------------------
    return render_template(
        "index1.html",
        file_content      = extracted_text,
        translated_content= translated_text,
        language          = LANGUAGE_CHOICES.get(detected_lang, detected_lang),
        output_language   = LANGUAGE_CHOICES.get(out_lang, out_lang),
        audio_filename    = audio_filename,
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=3000, debug=True)