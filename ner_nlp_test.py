# ner_nlp_test.py

import spacy
from bs4 import BeautifulSoup

# load html legal doc
def load_html_text(filename):
    # Open and read the HTML file
    with open(filename, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Parse the HTML
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Extract text from the parsed HTML
    text = soup.get_text(separator=' ', strip=True)
    return text

# chunk text so it fits into spacy
def chunk_text(text, chunk_size=1000000):
    # Initialize a list to store the chunks
    chunks = []
    
    # Calculate the number of chunks to create
    num_chunks = len(text) // chunk_size + 1
    
    # Create each chunk
    for i in range(num_chunks):
        # Calculate the starting index of the chunk
        start = i * chunk_size
        
        # Slice the text to create the chunk
        chunk = text[start:start + chunk_size]
        
        # Append the chunk to the list of chunks
        chunks.append(chunk)
    
    return chunks

# use spaCy to extract gov entities
def extract_gov_entities(text):
    # Process the text with spaCy
    doc = nlp(text)

    # Define entity labels that might correspond to government entities
    gov_entity_labels = {"ORG", "GPE"}

    # Initialize a set to store unique government entities
    gov_entities = set()

    # Iterate over the entities in the document
    for ent in doc.ents:
        # Check if the entity label is in our set of government entity labels
        if ent.label_ in gov_entity_labels:
            gov_entities.add(ent.text)

    return list(gov_entities)

# load the large English model
nlp = spacy.load("en_core_web_lg")

# load legal doc
html_text = load_html_text("sf_charter.html")

# get html text chunks
text_chunks = chunk_text(html_text)

gov_entities = []
for chunk in text_chunks:
    gov_entities += extract_gov_entities(chunk)
    #gov_entities = extract_gov_entities(html_text)

print(gov_entities)
