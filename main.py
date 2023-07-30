import requests
from bs4 import BeautifulSoup
import tiktoken
import pandas as pd

# Load unprocessed legislation data as a plain text file
url = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679&from=EN"
response = requests.get(url)

soup = BeautifulSoup(response.content, "html.parser")
legislation = soup.text

# Now we split the legislation into sections, each section starts with the world "Artikel"
search_document = legislation.split("HAVE ADOPTED THIS REGULATION:")
sections = search_document[1].split('\nArticle')
sections = ["Article" + section for section in sections]
sections = sections[1:]

# print(f'Amount of Articles: {len(sections)}')

section_titles = soup.find_all(class_='sti-art')
section_titles = [title.text for title in section_titles]

# print(f'Titles: {section_titles}')

# We can now parse each section using tiktoken, and calculate the amount of tokens per section
enc = tiktoken.encoding_for_model("gpt-4")
tokens_per_section = []

for section in sections:
    tokens = enc.encode(section)
    tokens_per_section.append(len(tokens))

print(f'Amount of Articles: {tokens_per_section}')

# Create a loop of 99 iterations
headings = []
countOfSections = len(sections)
for i in range(countOfSections):
    headings.append("Article " + str(i+1))

# Now let's load all the sections in a dataframe, appending the amount of tokens per section
df = pd.DataFrame()
df['title'] = section_titles
df['heading'] = headings
df['content'] = sections
df['tokens'] = tokens_per_section

# Write to CSV
df.to_csv('legislation.csv', index=False)

