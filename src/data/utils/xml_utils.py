from bs4 import BeautifulSoup

def crawlerXML(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    doc = BeautifulSoup(content)
    return doc
