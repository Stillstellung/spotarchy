import os
import time
import re
import cv2
import requests
import easyocr
import json

# Load conf file
with open("config.json") as config_file:
    config = json.load(config_file)

API_TOKEN = config["api_token"]

from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static'

# Create upload and output directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Define regex patterns for aircraft registrations
patterns = [
    r'N(?!.*[IO])[1-9]\d{0,4}[A-Z]{0,2}',  # US
    r'EI-[A-Z]{3}',                        # Ireland
    r'EJ-[A-Z]{4}',                        # Ireland VIP/Business
    r'A6-[A-Z]{3}',                        # UAE
    r'SU-[A-Z]{3}',                        # Egypt
    r'SU-[A-Z]{3}[A-Z0-9]{0,3}',           # Egypt (second range)
    r'HP-\d{4}[A-Z]{3}',                   # Panama
    r'OO-[A-Z]{3}',                        # Belgium (normal allocation)
    r'G-[A-Z]{4}',                         # UK
    r'YS-[A-Z]{3}',                        # El Salvador
    r'TI-[A-Z]{3}',                        # Costa Rica
    r'OE-[A-Z]{3}',                        # Austria (normal allocation)
    r'OE-[A-Z]{3}',                        # Austria (commercial allocation)
    r'JA\d{4}',                            # Japan
    r'JA\d{3}[A-Z]',                       # Japan
    r'JA\d{2}[A-Z]{2}',                    # Japan
    r'VT-[A-Z]{3}',                        # India
    r'F-[A-Z]{4}',                         # France
    r'B-\d{4}',                            # China
    r'C-F[A-Z]{3}',                        # Canada
    r'XA-[A-Z]{3}',                        # Mexico
    r'ET-[A-Z]{3}',                        # Ethiopia
    r'TF-[A-Z]{3}',                        # Iceland
    r'I-[A-Z]{4}',                         # Italy
    r'TC-[A-Z]{3}',                        # Turkey
    r'CS-[A-Z0-9]{3}',                     # Portugal
    r'HB-[A-Z]{3}',                        # Switzerland
    r'SE-[A-Z]{3}',                        # Sweden
    r'HZ-[A-Z]{3}',                        # Saudi Arabia
    r'CN-[A-Z]{3}',                        # Morocco
    r'D-[A-Z]{4}',                         # Germany
    r'HL\d{4}',                            # Korea
    r'PH-[A-Z]{3}',                        # Netherlands
]

# Combine all patterns into a single regex
combined_pattern = re.compile(r'^(' + '|'.join(patterns) + r')$')

def correct_registration(text, confidence):
    """Correct OCR errors in registrations based on context and confidence."""
    text = text.upper()  # Ensure uppercase for consistency

    # If the original text matches the regex, trust it and return unchanged
    if combined_pattern.match(text):
        return text

    # Apply corrections only if validation fails
    corrected_text = text.replace('I', '1').replace('O', '0')

    # For US registrations starting with 'N', apply further corrections
    if corrected_text.startswith('N'):
        corrected_text = corrected_text.replace('S', '5').replace('Z', '2')

    return corrected_text

def is_potential_registration(text):
    """Heuristic to filter out non-registration text."""
    # Registration-like strings are usually 4-7 characters long
    if not (4 <= len(text) <= 7):
        return False

    # Avoid obvious words or phrases (e.g., "AMERICAN", "EAGLE")
    if ' ' in text or text.isalpha():
        return False

    return True

def fetch_registration_details(registration):
    """Fetch details for a given aircraft registration from AeroDataBox."""
    api_token = API_TOKEN
    url = f"https://api.magicapi.dev/api/v1/aedbx/aerodatabox/aircrafts/Reg/{registration}?withImage=false&withRegistrations=false"
    headers = {
        "Authorization": f"Bearer {api_token}"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return {
            "typeName": data.get("typeName", "Unknown"),
            "airlineName": data.get("airlineName", "Unknown")
        }
    except requests.RequestException as e:
        print(f"Error fetching details for {registration}: {e}")
        return {"typeName": "Unknown", "airlineName": "Unknown"}

def process_image(image_path):
    """Process the image using EasyOCR and save output."""
    reader = easyocr.Reader(['en'], gpu=True)
    results = reader.readtext(image_path)

    print("\n=== Raw OCR Results ===")
    for bbox, text, confidence in results:
        print(f"Detected: '{text}' (Confidence: {confidence:.2f})")

    image = cv2.imread(image_path)
    matches = []

    print("\n=== Validation Results ===")
    for (bbox, text, confidence) in results:
        # Skip obvious non-registration text early
        if not is_potential_registration(text):
            print(f"🔄 Skipped: '{text}' (Not a potential registration)")
            continue

        # Correct the text and validate
        corrected_text = correct_registration(text, confidence)
        match = combined_pattern.match(corrected_text)

        print(f"Original: '{text}' | Corrected: '{corrected_text}' | Confidence: {confidence:.2f}")

        if match:
            print(f"✔️ Match: '{corrected_text}' is a valid registration")
            details = fetch_registration_details(corrected_text)
            print(f"   Details: Type: {details['typeName']}, Airline: {details['airlineName']}")
            matches.append((bbox, corrected_text, confidence, details))

            # Annotate image
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(image, f"{corrected_text} ({confidence:.2f})", 
                        (top_left[0], top_left[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            print(f"❌ Rejected: '{corrected_text}' does not match any pattern")

    output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.jpg')
    cv2.imwrite(output_path, image)
    return matches, output_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Start timer
        start_time = time.time()

        # Handle file upload
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Process the uploaded image
            results, output_path = process_image(filepath)

            # Calculate execution time
            execution_time = time.time() - start_time

            # Pass data to frontend
            return render_template(
                'index.html', 
                output_image=url_for('static', filename='output.jpg'), 
                results=results, 
                execution_time=round(execution_time, 2)
            )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
