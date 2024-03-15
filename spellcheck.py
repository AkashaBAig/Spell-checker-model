from collections import Counter
import re
import spacy
from symspellpy import SymSpell, Verbosity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import PyPDF2


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

# Path to uploaded PDF
pdf_path = 'setup.pdf'
text = extract_text_from_pdf(pdf_path)
print(text)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text() + ' '  # Extract text from each page
    return text


def preprocess_text(text):
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# New function to remove headers, footers, and TOC
def remove_headers_footers_toc(text):

    # Remove known headers or footers by pattern matching, if they have specific markers
    text = re.sub(r'HeaderPattern', '', text)
    text = re.sub(r'FooterPattern', '', text)
    # Remove TOC based on common identifiers or patterns
    text = re.sub(r'TOCPattern', '', text)
    # Example: Remove page numbers 
    text = re.sub(r'\n\d+\n', '\n', text)  # Remove standalone page numbers
    return text

# Function to simplify text further, placeholder for your actual simplification logic
def simplify_text(text):
    # Simplify text here (e.g., reducing complex sentences)
    # Placeholder for actual implementation
    return text

def prepare_data_for_training(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Additional preprocessing steps
    return text

def process_pdf(pdf_path):
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    print("Extracted Text:\n", text[:500], "...\n")  # Print the first 500 characters to keep the output manageable

    # Remove headers, footers, TOC
    text_no_headers_footers_toc = remove_headers_footers_toc(text)
    print("Text without Headers/Footers/TOC:\n", text_no_headers_footers_toc[:500], "...\n")

    # Preprocess text (stopwords removal)
    preprocessed_text = preprocess_text(text_no_headers_footers_toc)
    print("Preprocessed Text:\n", preprocessed_text[:500], "...\n")

    # Simplify text (placeholder for actual implementation)
    simplified_text = simplify_text(preprocessed_text)
    print("Simplified Text:\n", simplified_text[:500], "...\n")

    # Prepare data for training (final cleaning)
    training_data = prepare_data_for_training(simplified_text)
    print("Final Training Data:\n", training_data[:500], "...\n")

    return training_data


def count_word_frequencies(text):
    words = re.findall(r'\w+', text.lower())
    return Counter(words)

word_frequencies = count_word_frequencies(text)

def save_frequency_dictionary_to_file(freq_dict, file_path):
    with open(file_path, 'w') as file:
        for word, freq in freq_dict.items():
            file.write(f"{word} {freq}\n")

save_frequency_dictionary_to_file(word_frequencies, 'frequency_dictionary.txt')

# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

# Load your custom dictionary
custom_dictionary_path = "frequency_dictionary.txt"
sym_spell.load_dictionary(custom_dictionary_path, term_index=0, count_index=1)

def correct_spelling(text):
    # Look for suggestions
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    # Return the most likely suggestion
    return suggestions[0].term if suggestions else text

def process_query(query):
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Process query with spaCy
    doc = nlp(query)

    # Join tokens to reform the sentence (for context)
    joined_query = " ".join([token.text for token in doc])

    # Correct the entire sentence for context
    corrected_query = correct_spelling(joined_query)

    return corrected_query

#Example usage, can be removed during actual integration
incoming_query = "Veeer-Root makes no waranty of any kind with regard to this puliction  "
corrected_query = process_query(incoming_query)
print(corrected_query)