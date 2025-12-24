# extracts the text from the documents (.pdf, .docx, .txt)

import fitz #pymupdf
from docx import Document
from pathlib import Path

class TextExtractor:

    def extract(self, file_path:str)-> str:
        #extract the text from a file with respect to its extension
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension == ".pdf":
            return self._extract_pdf(file_path)
        elif extension == ".docx":
            return self._extract_docx(file_path)
        elif extension == ".txt":
            return self._extract_txt(file_path)
        else:
            raise ValueError(f"Formant not supported {extension}")
        
    def _extract_pdf(self, file_path:str)->str:
        #extract text from a .pdf file
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text +=page.get_text()
        doc.close()
        return text

    def _extract_docx(self, file_path:str)->str:
        doc=Document(file_path)
        text="" 
        for paragraph in doc.paragraphs:
            text+=paragraph.text+"\n"
        return text
    
    def _extract_txt(self, file_path:str)->str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
        
