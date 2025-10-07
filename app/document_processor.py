import PyPDF2
from docx import Document
import pandas as pd
import openpyxl
from pathlib import Path
from typing import Dict, List, Any

class DocumentProcessor:
    """Handles extraction of text and data from various file formats"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.xlsx', '.xls', '.csv']
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Main method to process any supported file
        Returns: Dictionary with extracted content
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self._process_pdf(file_path)
        elif extension == '.docx':
            return self._process_docx(file_path)
        elif extension in ['.xlsx', '.xls']:
            return self._process_excel(file_path)
        elif extension == '.csv':
            return self._process_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _process_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from PDF"""
        print(f"📄 Reading PDF: {file_path.name}")
        
        text_content = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                text_content.append(text)
        
        return {
            'type': 'pdf',
            'filename': file_path.name,
            'pages': num_pages,
            'content': '\n'.join(text_content),
            'raw_pages': text_content
        }
    
    def _process_docx(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from Word document"""
        print(f"📝 Reading DOCX: {file_path.name}")
        
        doc = Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        
        return {
            'type': 'docx',
            'filename': file_path.name,
            'paragraphs': len(paragraphs),
            'content': '\n'.join(paragraphs),
            'raw_paragraphs': paragraphs
        }
    
    def _process_excel(self, file_path: Path) -> Dict[str, Any]:
        """Extract data from Excel file"""
        print(f"📊 Reading Excel: {file_path.name}")
        
        # Read all sheets
        excel_file = pd.ExcelFile(file_path)
        sheets_data = {}
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            sheets_data[sheet_name] = df
        
        return {
            'type': 'excel',
            'filename': file_path.name,
            'sheets': list(sheets_data.keys()),
            'data': sheets_data,
            'summary': self._summarize_excel_data(sheets_data)
        }
    
    def _process_csv(self, file_path: Path) -> Dict[str, Any]:
        """Extract data from CSV file"""
        print(f"📈 Reading CSV: {file_path.name}")
        
        df = pd.read_csv(file_path)
        
        return {
            'type': 'csv',
            'filename': file_path.name,
            'rows': len(df),
            'columns': list(df.columns),
            'data': df,
            'summary': df.describe().to_dict()
        }
    
    def _summarize_excel_data(self, sheets_data: Dict) -> str:
        """Create a text summary of Excel data"""
        summary = []
        for sheet_name, df in sheets_data.items():
            summary.append(f"Sheet '{sheet_name}': {len(df)} rows, {len(df.columns)} columns")
            summary.append(f"Columns: {', '.join(df.columns)}")
        return '\n'.join(summary)