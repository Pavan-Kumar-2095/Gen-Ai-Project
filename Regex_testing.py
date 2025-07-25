import os
import re
import string
import fitz  # PyMuPDF
import mammoth
from mailparser import parse_from_bytes
from transformers import AutoTokenizer
from nltk.corpus import stopwords

# NLP/Tokenizer setup
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Custom cleaning setup
company_name = "Bajaj Allianz General Insurance Company Limited"
abbreviation = "Insurer"

patterns_to_remove = [
    r"UIN- BAJHLIP\d+V\d+",
    r"Bajaj Allianz.*?Pune\s*-\s*411\s*006\.",
    r"Global Health Care/ Policy Wordings/Page \d+",
    r"Call at:.*?\(Toll Free No\.\)",
    r"For more details, log on to:.*",
]


def extract_text_from_file(file_path: str) -> str:
    ext = file_path.split('.')[-1].lower()

    if ext == "pdf":
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text() for page in doc])
        return text

    elif ext == "docx":
        result = mammoth.extract_raw_text({'path': file_path})
        return result.value

    elif ext == "eml":
        with open(file_path, "rb") as f:
            parsed = parse_from_bytes(f.read())
            return parsed.body or (parsed.text_plain[0] if parsed.text_plain else "")

    else:
        raise ValueError("Unsupported file format.")


def clean_text(text: str) -> str:
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = text.replace(company_name, abbreviation)

    # Collapse bullet points and list items
    text = re.sub(r"\b[iivxl]+\.\s+", "• ", text)   # Roman numerals
    text = re.sub(r"\b[a-zA-Z]\)\s+", "• ", text)   # Alphabet bullets
    text = re.sub(r"\n\s*•", " •", text)

    # Normalize whitespace
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunk = text[i:i + size]
        if chunk:
            chunks.append(chunk)
    return chunks


def main():
    file_path = input("Enter path to your document (.pdf/.docx/.eml): ").strip()

    if not os.path.exists(file_path):
        print("❌ File not found.")
        return

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    try:
        print("\n📥 Extracting text...")
        raw_text = extract_text_from_file(file_path)
        print(f"📝 Raw text length: {len(raw_text)} characters")

        print("\n🔪 Chunking raw text...")
        raw_chunks = chunk_text(raw_text)
        print(f"📦 Total Chunks: {len(raw_chunks)}\n")

        total_tokens_before = 0
        total_tokens_after = 0

        for i, chunk in enumerate(raw_chunks[:]):  # can limit to first N chunks
            tokens_before = len(tokenizer.tokenize(chunk))
            cleaned_chunk = clean_text(chunk)
            tokens_after = len(tokenizer.tokenize(cleaned_chunk))

            total_tokens_before += tokens_before
            total_tokens_after += tokens_after

        print("===========================================")
        print(f"Total tokens before cleaning:     {total_tokens_before}")
        print(f"Total tokens after cleaning:      {total_tokens_after}")
        print(f"Total tokens saved:               {total_tokens_before - total_tokens_after}")
        print(f"Percentage tokens saved:          {(total_tokens_before - total_tokens_after) / total_tokens_before * 100:.2f}%")
        print("===========================================")

    except Exception as e:
        print("❌ Error:", str(e))


if __name__ == "__main__":
    main()
