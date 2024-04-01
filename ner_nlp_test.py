# ner_nlp_test.py

import spacy
from bs4 import BeautifulSoup
import csv
import os
from datetime import datetime
import time
import resource

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


def write_to_csv(filename, new_data):
    # Generate a timestamp for the header
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Check if the file exists
    file_exists = os.path.isfile(filename)

    if file_exists:
        # Read existing data
        with open(filename, 'r', newline='') as file:
            reader = csv.reader(file)
            existing_data = list(reader)

        # Prepare new column with timestamp header and new_data
        #new_column = [timestamp] + new_data
        new_column_data = [timestamp] + new_data
        
        # Determine the maximum length of the existing data
        max_length = max(len(row) for row in existing_data)
        
        # Append new_column_data to existing_data, adjusting for any length differences
        for i, new_data in enumerate(new_column_data):
            if i < len(existing_data):
                # Append new data to existing rows
                existing_data[i].append(new_data)
            else:
                # If new_column_data is longer than existing_data, add new rows
                existing_data.append([''] * max_length + [new_data])

        # If existing_data is longer than new_column_data, fill in the blanks
        for i in range(len(new_column_data), len(existing_data)):
            existing_data[i].append('')

        # Write the updated data back to the file
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(existing_data)

    else:
        # Create a new file and write the data with a timestamp header
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp])  # Write the header
            for item in new_data:
                writer.writerow([item])  # Write each item in a new row under the timestamp header

# start timer
start_time = time.time()

# load the large English model
nlp = spacy.load("en_core_web_lg")

# load legal doc
#html_text = load_html_text("sf_charter.html")
html_text = load_html_text("sf_charter_sec234only.html")

# get html text chunks
text_chunks = chunk_text(html_text)

# spaCy NER on each chunk
gov_entities = []
for chunk in text_chunks:
    chunk_entities = extract_gov_entities(chunk)
    print(f"Identified {len(chunk_entities)} entities in chunk")

    gov_entities += chunk_entities

# write to file
filename = 'gov_entities.csv'
write_to_csv(filename, gov_entities)

# end timer, get execution time
end_time = time.time()
execution_time = end_time - start_time
# gather memory usage (Unix-based systems)
memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
memory_usage_mb = memory_usage / (1024*1024)

print(f"Identified {len(gov_entities)} entities")
print(f"Execution time: {execution_time} seconds")
print(f"Peak Memory Usage: {memory_usage_mb} MB")
