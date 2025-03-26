import pytesseract
from PIL import Image
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import pandas as pd
from pdf2image import convert_from_path
import os
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('all')

# Path to Tesseract executable (update this based on your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows example

# Step 1: OCR - Extract text from PDF or image
def extract_text_from_file(file_path):
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file type
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.pdf':
            # Convert PDF to images
            images = convert_from_path(file_path)
            text = ""
            for i, image in enumerate(images):
                # Convert image to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                # Perform OCR on each page
                page_text = pytesseract.image_to_string(image, lang='eng')
                text += f"\n--- Page {i+1} ---\n{page_text}"
            return text

        elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            # Open the image
            image = Image.open(file_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Perform OCR
            text = pytesseract.image_to_string(image, lang='eng')
            return text

        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    except Exception as e:
        print(f"Error processing file: {e}")
        raise

# Step 2: Preprocess the extracted text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    # Remove stopwords and tokenize words
    stop_words = set(stopwords.words('english'))
    tokens = [word_tokenize(sent) for sent in sentences]
    tokens = [[word for word in sent if word not in stop_words and word.isalnum()] for sent in tokens]
    return sentences, tokens

# Step 3: Classify Request Type and Sub-Request Type
def classify_request_type(text, tokens):
    # Define predefined request types and keywords (simulating LLM behavior)
    request_types = {
        "principal_payment": ["principal", "payment", "repay", "balance"],
        "loan_modification": ["modify", "modification", "term", "sofr"],
        "interest_payment": ["interest", "rate", "payment"],
        "general_inquiry": ["inquiry", "question", "information"]
    }

    # Score each request type based on keyword matches
    scores = {req_type: 0 for req_type in request_types}
    for req_type, keywords in request_types.items():
        for keyword in keywords:
            if keyword in text:
                scores[req_type] += 1

    # Determine primary request type (highest score)
    primary_request = max(scores, key=scores.get)
    confidence = scores[primary_request] / sum(scores.values()) if sum(scores.values()) > 0 else 0

    # Sub-request type (for principal_payment, check if it's a partial or full payment)
    sub_request = "full_payment" if "total of usd" in text else "partial_payment"

    return primary_request, sub_request, confidence

# Step 4: Extract Key Fields
def extract_fields(text):
    fields = {}

    # Extract Deal Name (e.g., CANTOR FITZGERALD LP USD 425MM MAR22)
    deal_pattern = r"re:\s*(cantor fitzgerald lp usd \d+mm mar\d+)"
    deal_match = re.search(deal_pattern, text)
    fields["deal_name"] = deal_match.group(1) if deal_match else "Not Found"

    # Extract Amount (e.g., USD 20,000,000.00)
    amount_pattern = r"usd\s*(\d{1,3}(?:,\d{3})*\.\d{2})"
    amount_match = re.search(amount_pattern, text)
    fields["amount"] = amount_match.group(1) if amount_match else "Not Found"

    # Extract Date (e.g., 20-Nov-2023)
    date_pattern = r"\d{1,2}-[a-z]{3}-\d{4}"
    date_match = re.search(date_pattern, text)
    fields["effective_date"] = date_match.group(0) if date_match else "Not Found"

    return fields

# Step 5: Handle Multi-Request and Priority
def handle_multi_request(text, primary_request):
    # Check for secondary requests (e.g., mentions of "interest" alongside principal payment)
    secondary_requests = []
    if "interest" in text and primary_request != "interest_payment":
        secondary_requests.append("interest_payment")
    if "modify" in text and primary_request != "loan_modification":
        secondary_requests.append("loan_modification")
    
    return secondary_requests

# Step 6: Duplicate Detection
def detect_duplicates(text):
    # Check for reference number or email thread indicators
    ref_pattern = r"reference\s*:\s*(cantor fitzgerald lp usd \d+mm mar\d+)"
    ref_match = re.search(ref_pattern, text)
    if ref_match:
        return True, "Duplicate detected based on reference number: " + ref_match.group(1)
    return False, "No duplicate detected"

# Step 7: Main Pipeline
def process_document(file_path, email_content=None):
    # Extract text from the document (PDF or image)
    doc_text = extract_text_from_file(file_path)

    # If email content is provided, prioritize it for request type identification
    if email_content:
        combined_text = email_content + "\n\n--- Attachment ---\n" + doc_text
    else:
        combined_text = doc_text

    # Preprocess text
    sentences, tokens = preprocess_text(combined_text)

    # Classify request type (prioritize email content if provided)
    if email_content:
        email_sentences, email_tokens = preprocess_text(email_content.lower())
        primary_request, sub_request, confidence = classify_request_type(email_content.lower(), email_tokens)
    else:
        primary_request, sub_request, confidence = classify_request_type(combined_text, tokens)

    # Extract fields (use combined text to ensure numerical fields from attachments are captured)
    fields = extract_fields(combined_text)

    # Handle multi-request
    secondary_requests = handle_multi_request(combined_text, primary_request)

    # Detect duplicates
    is_duplicate, duplicate_reason = detect_duplicates(combined_text)

    # Prepare output
    output = {
        "Primary Request Type": primary_request,
        "Sub Request Type": sub_request,
        "Confidence Score": confidence,
        "Secondary Requests": secondary_requests,
        "Extracted Fields": fields,
        "Duplicate Flag": is_duplicate,
        "Duplicate Reason": duplicate_reason
    }

    return output

# Step 8: Test the pipeline
if __name__ == "__main__":
    # Print current working directory for debugging
    print(f"Current working directory: {os.getcwd()}")

    # Path to the PDF document (update this to your PDF file)
    file_path = "test_email.pdf"  # Replace with the actual path to your PDF
    print(f"Attempting to process file: {file_path}")

    # Optional: Simulate email content (if the PDF is an attachment)
    email_content = """
    Subject: Request for Principal Payment
    Dear Team,
    Please process the attached loan servicing request for a principal payment.
    Regards,
    John Doe
    """

    try:
        # Process the document
        result = process_document(file_path, email_content)

        # Display results in a structured format
        df = pd.DataFrame([result])
        print("\nStructured Output:")
        print(df)

        # Pretty print the output
        print("\nDetailed Output:")
        for key, value in result.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"Error in main execution: {e}")