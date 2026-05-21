import pytesseract
import cv2
import os
import glob

# Set path if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Find the latest image in the outputfolder
output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputfolder")
os.makedirs(output_folder, exist_ok=True)
list_of_files = glob.glob(os.path.join(output_folder, "*.png"))

if not list_of_files:
    print("No images found in outputfolder.")
else:
    # Get the most recently modified file (the latest image)
    latest_file = max(list_of_files, key=os.path.getmtime)
    print(f"Reading image: {latest_file}")
    
    # Read image
    img = cv2.imread(latest_file)
    
    # --- ADVANCED PREPROCESSING ---
    # 1. Upscale the image to make small text more readable for the OCR engine
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Noise removal using Bilateral Filter (removes noise while keeping text edges sharp)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 4. Otsu's Thresholding (Binarization: makes background perfectly white and text perfectly black)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Extract text using advanced Tesseract configuration
    # OEM=3 (Default Engine), PSM=6 (Assume a single uniform block of text)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    
    print("\n--- EXTRACTED TEXT ---")
    print(text)