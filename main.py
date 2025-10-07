import os
from pathlib import Path
from typing import List, Dict, Any
import yaml
from dotenv import load_dotenv
import openai

# Import our custom modules
from app.document_processor import DocumentProcessor
from app.web_searcher import WebSearcher
from app.data_analyzer import DataAnalyzer
from chart_generator import ChartGenerator
from app.report_writer import ReportWriter

class AutomatedReportGenerator:
    """
    Main orchestrator for the automated report generation system
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize all components"""
        print("🚀 Initializing Automated Report Generator...")
        
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize OpenAI
        openai.api_key = os.getenv('OPENAI_API_KEY') or self.config['openai']['api_key']
        
        # Initialize components
        self.doc_processor = DocumentProcessor()
        self.web_searcher = WebSearcher(max_results=self.config['search']['max_results'])
        self.data_analyzer = DataAnalyzer()
        self.chart_generator = ChartGenerator()
        self.report_writer = ReportWriter()
        
        print("✅ All components initialized successfully!\n")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print("⚠️  Config file not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'openai': {
                'model': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 2000
            },
            'search': {
                'max_results': 5
            },
            'report': {
                'default_format': 'pdf',
                'include_charts': True,
                'include_references': True,
                'sections': [
                    'Executive Summary',
                    'Key Findings',
                    'Data Analysis',
                    'Visualizations',
                    'Recommendations',
                    'References'
                ]
            }
        }
    
    def generate_report(
        self,
        topic: str,
        document_paths: List[str] = None,
        web_search: bool = True,
        output_format: str = 'pdf'
    ) -> str:
        """
        Main method to generate a complete report
        
        Args:
            topic: The main topic/question for the report
            document_paths: List of paths to documents to analyze
            web_search: Whether to search the web for additional info
            output_format: Output format (pdf, docx, html)
        
        Returns:
            Path to the generated report
        """
        print(f"\n{'='*60}")
        print(f"🎯 GENERATING REPORT: {topic}")
        print(f"{'='*60}\n")
        
        # Step 1: Process documents
        document_data = []
        if document_paths:
            print("📁 STEP 1: Processing Documents")
            print("-" * 60)
            for doc_path in document_paths:
                try:
                    data = self.doc_processor.process_file(doc_path)
                    document_data.append(data)
                except Exception as e:
                    print(f"❌ Error processing {doc_path}: {e}")
            print(f"✅ Processed {len(document_data)} documents\n")
        
        # Step 2: Web search
        web_results = []
        if web_search:
            print("🌐 STEP 2: Searching Web")
            print("-" * 60)
            web_results = self.web_searcher.search(topic)
            print(f"✅ Found {len(web_results)} web sources\n")
        
        # Step 3: Analyze data and generate insights using AI
        print("🧠 STEP 3: Analyzing Data with AI")
        print("-" * 60)
        insights = self._generate_insights(topic, document_data, web_results)
        print("✅ Analysis complete\n")
        
        # Step 4: Generate charts
        charts = []
        print("📊 STEP 4: Generating Visualizations")
        print("-" * 60)
        charts = self._generate_charts(document_data, insights)
        print(f"✅ Created {len(charts)} charts\n")
        
        # Step 5: Create report
        print("📝 STEP 5: Compiling Report")
        print("-" * 60)
        report_path = self._create_final_report(topic, insights, charts, web_results)
        
        print(f"\n{'='*60}")
        print(f"✅ REPORT GENERATED SUCCESSFULLY!")
        print(f"📄 Location: {report_path}")
        print(f"{'='*60}\n")
        
        return report_path
    
    def _generate_insights(
        self,
        topic: str,
        document_data: List[Dict],
        web_results: List[Dict]
    ) -> Dict[str, Any]:
        """
        Use LLM to generate insights from all collected data
        """
        # Prepare context for LLM
        context = self._prepare_context(document_data, web_results)
        
        # Generate different sections using AI
        insights = {
            'executive_summary': self._generate_section(
                topic, context, "executive summary", 200
            ),
            'key_findings': self._generate_section(
                topic, context, "key findings as bullet points", 300
            ),
            'analysis': self._generate_section(
                topic, context, "detailed analysis", 500
            ),
            'recommendations': self._generate_section(
                topic, context, "actionable recommendations", 300
            )
        }
        
        return insights
    
    def _prepare_context(
        self,
        document_data: List[Dict],
        web_results: List[Dict]
    ) -> str:
        """Prepare context string from all sources"""
        context_parts = []
        
        # Add document content
        for i, doc in enumerate(document_data, 1):
            context_parts.append(f"\n--- Document {i}: {doc['filename']} ---")
            if 'content' in doc:
                # Limit content length
                content = doc['content'][:3000]
                context_parts.append(content)
            elif 'summary' in doc:
                context_parts.append(str(doc['summary']))
        
        # Add web search results
        if web_results:
            context_parts.append("\n--- Web Search Results ---")
            for result in web_results:
                context_parts.append(f"\nTitle: {result['title']}")
                context_parts.append(f"Source: {result['url']}")
                context_parts.append(f"Content: {result['snippet']}")
        
        return '\n'.join(context_parts)
    
    def _generate_section(
        self,
        topic: str,
        context: str,
        section_type: str,
        max_words: int
    ) -> str:
        """
        Generate a specific section using OpenAI
        """
        try:
            prompt = f"""Based on the following context, create a {section_type} for a report about "{topic}".

Context:
{context[:10000]}  # Limit context to avoid token limits

Instructions:
- Be concise and professional
- Maximum {max_words} words
- Focus on the most important information
- Use clear, accessible language

{section_type.upper()}:"""

            response = openai.chat.completions.create(
                model=self.config['openai']['model'],
                messages=[
                    {"role": "system", "content": "You are an expert analyst creating professional business reports."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config['openai']['temperature'],
                max_tokens=self.config['openai']['max_tokens']
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Error generating {section_type}: {e}")
            return f"Error generating {section_type}. Please check your API configuration."
    
    def _generate_charts(
        self,
        document_data: List[Dict],
        insights: Dict
    ) -> List[str]:
        """Generate charts from data)