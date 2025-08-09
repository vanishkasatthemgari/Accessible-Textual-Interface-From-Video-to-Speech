from PIL import Image
import pytesseract

# Optional: Set the path to tesseract if not in PATH
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Load the stitched group image you want to test
img = Image.open("static/stitched_groups/group_3.jpg").convert("L")
img = img.resize((img.width * 2, img.height * 2), Image.BICUBIC)

# Run Tesseract OCR
text = pytesseract.image_to_string(img, lang="eng", config="--psm 11")
print("\n📝 OCR Output:\n")
print(text.strip())
