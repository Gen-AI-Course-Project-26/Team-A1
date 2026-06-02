import pdfplumber
import re

with pdfplumber.open('data/raw/constitution-of-india.pdf') as pdf:
    text = ''
    for page in pdf.pages[30:35]:
        text += page.extract_text() or ''

lines = text.split('\n')
for line in lines:
    line = line.strip()
    if line:
        print(repr(line[:90]))