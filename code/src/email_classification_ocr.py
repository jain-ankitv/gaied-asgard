import os
import json
import email
import base64
import fitz  # PyMuPDF for PDF processing
import openai
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple

# Initialize FastAPI
app = FastAPI()

# Load API keys (ensure you set these as environment variables)
openai.api_key = os.getenv("OPENAI_API_KEY")

class EmailRequest(BaseModel):
    sender: str
    subject: str
    body: str
    attachments: List[UploadFile] = []
    request_definitions: Dict[str, List[str]]
    extraction_rules: Dict[str, str]

@app.post("/process_email/")
async def process_email(request: EmailRequest):
    """Process incoming emails for classification and data extraction."""
    extracted_text = request.body
    
    # Process attachments
    for attachment in request.attachments:
        content = await attachment.read()
        extracted_text += extract_text_from_attachment(attachment.filename, content)
    
    # Classify email intent
    request_type, sub_request_type, confidence = await classify_email(extracted_text, request.request_definitions)
    
    # Extract relevant fields
    extracted_data = await extract_fields(extracted_text, request.extraction_rules)
    
    # Check for duplicates
    is_duplicate, reason = check_duplicate(request.sender, request.subject, extracted_text)
    
    return {
        "request_type": request_type,
        "sub_request_type": sub_request_type,
        "confidence": confidence,
        "extracted_data": extracted_data,
        "duplicate": is_duplicate,
        "duplicate_reason": reason,
    }

def extract_text_from_attachment(filename: str, content: bytes) -> str:
    """Extract text from attachments (PDFs, text files)."""
    extracted_text = ""
    if filename.lower().endswith(".pdf"):
        with fitz.open(stream=content, filetype="pdf") as doc:
            for page in doc:
                extracted_text += page.get_text("text") + "\n"
    elif filename.lower().endswith(".txt"):
        extracted_text = content.decode("utf-8")
    return extracted_text

async def classify_email(text: str, request_definitions: Dict[str, List[str]]) -> Tuple[str, str, float]:
    """Use LLMs to classify email request type and sub-type."""
    prompt = f"""
    Classify the following email into one of the predefined categories:
    {text}
    Categories:
    {json.dumps(request_definitions, indent=2)}
    Respond in JSON format: {{"request_type": ..., "sub_request_type": ..., "confidence": ...}}
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}]
    )
    classification = json.loads(response['choices'][0]['message']['content'])
    return classification.get("request_type"), classification.get("sub_request_type"), classification.get("confidence", 0.0)

async def extract_fields(text: str, extraction_rules: Dict[str, str]) -> Dict[str, str]:
    """Extract fields from email text using LLM-based parsing."""
    prompt = f"""
    Extract the following fields from the given text:
    {json.dumps(extraction_rules, indent=2)}
    Text:
    {text}
    Respond in JSON format with extracted values.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}]
    )
    return json.loads(response['choices'][0]['message']['content'])

def check_duplicate(sender: str, subject: str, body: str) -> Tuple[bool, str]:
    """Basic duplicate detection based on sender, subject, and body similarity."""
    # In a real system, use a database and similarity matching
    hash_key = hash((sender, subject, body))
    if hash_key in seen_emails:
        return True, "Email is a duplicate based on sender, subject, and content."
    seen_emails.add(hash_key)
    return False, "Unique email."

# Temporary storage for seen emails
seen_emails = set()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)