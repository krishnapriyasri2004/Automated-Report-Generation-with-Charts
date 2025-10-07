from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import textwrap

class ReportWriter:
    """Generates professional PDF reports"""
    
    def __init__(self, output_dir: str = "output/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a365d'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2d3748'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=12,
            leading=16
        ))
    
    def create_report(self, report_data: Dict[str, Any], output_filename: str = None) -> str:
        """
        Create a complete PDF report
        
        report_data should contain:
        - title: str
        - author: str (optional)
        - sections: List[Dict] with 'heading' and 'content'
        - charts: List[str] - paths to chart images
        - references: List[str] (optional)
        """
        
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"report_{timestamp}.pdf"
        
        output_path = self.output_dir / output_filename
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Add title page
        elements.extend(self._create_title_page(report_data))
        elements.append(PageBreak())
        
        # Add table of contents (optional)
        if report_data.get('include_toc', False):
            elements.extend(self._create_toc(report_data))
            elements.append(PageBreak())
        
        # Add sections
        for section in report_data.get('sections', []):
            elements.extend(self._create_section(section))
        
        # Add charts
        if report_data.get('charts'):
            elements.append(Paragraph("Data Visualizations", self.styles['SectionHeader']))
            elements.append(Spacer(1, 12))
            for chart_path in report_data['charts']:
                elements.extend(self._add_chart(chart_path))
        
        # Add references
        if report_data.get('references'):
            elements.append(PageBreak())
            elements.extend(self._create_references(report_data['references']))
        
        # Build PDF
        doc.build(elements)
        
        print(f"📄 Report created: {output_path}")
        return str(output_path)
    
    def _create_title_page(self, report_data: Dict) -> List:
        """Create title page elements"""
        elements = []
        
        # Add some space at top
        elements.append(Spacer(1, 2*inch))
        
        # Title
        title = report_data.get('title', 'Automated Report')
        elements.append(Paragraph(title, self.styles['CustomTitle']))
        elements.append(Spacer(1, 0.5*inch))
        
        # Subtitle/description
        if 'subtitle' in report_data:
            subtitle_style = ParagraphStyle(
                name='Subtitle',
                parent=self.styles['Normal'],
                fontSize=14,
                textColor=colors.grey,
                alignment=TA_CENTER
            )
            elements.append(Paragraph(report_data['subtitle'], subtitle_style))
            elements.append(Spacer(1, 0.3*inch))
        
        # Metadata table
        metadata = [
            ['Generated:', datetime.now().strftime("%B %d, %Y at %I:%M %p")],
            ['Author:', report_data.get('author', 'AI Report Generator')],
        ]
        
        if 'topic' in report_data:
            metadata.append(['Topic:', report_data['topic']])
        
        table = Table(metadata, colWidths=[1.5*inch, 4*inch])
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(Spacer(1, 1*inch))
        elements.append(table)
        
        return elements
    
    def _create_section(self, section: Dict) -> List:
        """Create a report section"""
        elements = []
        
        # Section heading
        heading = section.get('heading', 'Section')
        elements.append(Paragraph(heading, self.styles['SectionHeader']))
        elements.append(Spacer(1, 12))
        
        # Section content
        content = section.get('content', '')
        
        # Split content into paragraphs
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            if para.strip():
                elements.append(Paragraph(para.strip(), self.styles['BodyText']))
                elements.append(Spacer(1, 6))
        
        # Add bullet points if provided
        if 'bullets' in section:
            for bullet in section['bullets']:
                bullet_style = ParagraphStyle(
                    name='Bullet',
                    parent=self.styles['Normal'],
                    leftIndent=20,
                    bulletIndent=10,
                    fontSize=11
                )
                elements.append(Paragraph(f"• {bullet}", bullet_style))
                elements.append(Spacer(1, 6))
        
        # Add data table if provided
        if 'table' in section:
            elements.append(Spacer(1, 12))
            elements.extend(self._create_table(section['table']))
        
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _add_chart(self, chart_path: str) -> List:
        """Add a chart image to the report"""
        elements = []
        
        try:
            img = Image(chart_path, width=5*inch, height=3*inch)
            elements.append(img)
            elements.append(Spacer(1, 12))
            
            # Add caption
            caption = Path(chart_path).stem.replace('_', ' ').title()
            caption_style = ParagraphStyle(
                name='Caption',
                parent=self.styles['Normal'],
                fontSize=9,
                textColor=colors.grey,
                alignment=TA_CENTER,
                italic=True
            )
            elements.append(Paragraph(f"Figure: {caption}", caption_style))
            elements.append(Spacer(1, 20))
            
        except Exception as e:
            print(f"❌ Error adding chart {chart_path}: {e}")
        
        return elements
    
    def _create_table(self, table_data: List[List]) -> List:
        """Create a formatted table"""
        elements = []
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a5568')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        
        elements.append(table)
        
        return elements
    
    def _create_references(self, references: List[str]) -> List:
        """Create references section"""
        elements = []
        
        elements.append(Paragraph("References", self.styles['SectionHeader']))
        elements.append(Spacer(1, 12))
        
        for i, ref in enumerate(references, 1):
            ref_style = ParagraphStyle(
                name='Reference',
                parent=self.styles['Normal'],
                fontSize=10,
                leftIndent=20,
                firstLineIndent=-20,
                spaceAfter=8
            )
            elements.append(Paragraph(f"[{i}] {ref}", ref_style))
        
        return elements
    
    def _create_toc(self, report_data: Dict) -> List:
        """Create table of contents"""
        elements = []
        
        elements.append(Paragraph("Table of Contents", self.styles['SectionHeader']))
        elements.append(Spacer(1, 12))
        
        for i, section in enumerate(report_data.get('sections', []), 1):
            heading = section.get('heading', f'Section {i}')
            toc_style = ParagraphStyle(
                name='TOC',
                parent=self.styles['Normal'],
                fontSize=11,
                leftIndent=20,
                spaceAfter=6
            )
            elements.append(Paragraph(f"{i}. {heading}", toc_style))
        
        return elements