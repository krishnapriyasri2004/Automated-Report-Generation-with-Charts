#!/usr/bin/env python3
"""
AUTOMATED REPORT GENERATOR - SIMPLIFIED STARTER
================================================
This is a simplified version that works WITHOUT:
- OpenAI API
- Complex dependencies
- Configuration files

Just run: python simple_starter.py

This will demonstrate the core concepts with minimal setup.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Check if basic libraries are available
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.units import inch
    print("✅ All required libraries found!")
except ImportError as e:
    print(f"❌ Missing library: {e}")
    print("\n📦 Please install requirements:")
    print("pip install pandas matplotlib reportlab")
    sys.exit(1)


class SimpleReportGenerator:
    """Simplified report generator without external APIs"""
    
    def __init__(self):
        self.output_dir = Path("simple_output")
        self.output_dir.mkdir(exist_ok=True)
        print(f"📁 Output directory: {self.output_dir.absolute()}")
    
    def create_sample_data(self):
        """Create sample data for demonstration"""
        print("\n📊 Creating sample data...")
        
        # Sample sales data
        data = {
            'Month': ['January', 'February', 'March', 'April', 'May', 'June'],
            'Revenue': [45000, 52000, 48000, 61000, 58000, 67000],
            'Expenses': [30000, 33000, 31000, 38000, 36000, 41000],
            'Customers': [120, 145, 135, 170, 165, 190]
        }
        
        df = pd.DataFrame(data)
        df['Profit'] = df['Revenue'] - df['Expenses']
        
        print(f"✅ Generated {len(df)} months of data")
        return df
    
    def create_charts(self, df):
        """Create simple charts"""
        print("\n📈 Generating charts...")
        chart_paths = []
        
        # Chart 1: Revenue vs Expenses
        plt.figure(figsize=(10, 6))
        plt.plot(df['Month'], df['Revenue'], marker='o', linewidth=2, label='Revenue', color='green')
        plt.plot(df['Month'], df['Expenses'], marker='s', linewidth=2, label='Expenses', color='red')
        plt.title('Monthly Revenue vs Expenses', fontsize=16, fontweight='bold')
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Amount ($)', fontsize=12)
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        chart1_path = self.output_dir / "chart1_revenue_expenses.png"
        plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths.append(str(chart1_path))
        print(f"  ✅ Created: {chart1_path.name}")
        
        # Chart 2: Profit Trend
        plt.figure(figsize=(10, 6))
        plt.bar(df['Month'], df['Profit'], color='steelblue', alpha=0.7)
        plt.title('Monthly Profit', fontsize=16, fontweight='bold')
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Profit ($)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        chart2_path = self.output_dir / "chart2_profit.png"
        plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths.append(str(chart2_path))
        print(f"  ✅ Created: {chart2_path.name}")
        
        # Chart 3: Customer Growth
        plt.figure(figsize=(10, 6))
        plt.plot(df['Month'], df['Customers'], marker='D', linewidth=2.5, 
                color='purple', markersize=8)
        plt.fill_between(range(len(df)), df['Customers'], alpha=0.3, color='purple')
        plt.title('Customer Growth', fontsize=16, fontweight='bold')
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Number of Customers', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        chart3_path = self.output_dir / "chart3_customers.png"
        plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
        plt.close()
        chart_paths.append(str(chart3_path))
        print(f"  ✅ Created: {chart3_path.name}")
        
        return chart_paths
    
    def analyze_data(self, df):
        """Simple data analysis without AI"""
        print("\n🧠 Analyzing data...")
        
        analysis = {}
        
        # Calculate key metrics
        analysis['total_revenue'] = df['Revenue'].sum()
        analysis['total_expenses'] = df['Expenses'].sum()
        analysis['total_profit'] = df['Profit'].sum()
        analysis['avg_monthly_revenue'] = df['Revenue'].mean()
        analysis['revenue_growth'] = ((df['Revenue'].iloc[-1] - df['Revenue'].iloc[0]) / df['Revenue'].iloc[0]) * 100
        analysis['customer_growth'] = df['Customers'].iloc[-1] - df['Customers'].iloc[0]
        analysis['best_month'] = df.loc[df['Profit'].idxmax(), 'Month']
        analysis['best_profit'] = df['Profit'].max()
        
        print("  ✅ Analysis complete")
        return analysis
    
    def generate_insights(self, analysis):
        """Generate simple insights without AI"""
        insights = {
            'executive_summary': f"""
This report provides a comprehensive analysis of business performance over a 6-month period.
Total revenue reached ${analysis['total_revenue']:,.0f} with a total profit of ${analysis['total_profit']:,.0f}.
The business showed a revenue growth of {analysis['revenue_growth']:.1f}% during this period.
Customer acquisition increased by {analysis['customer_growth']} customers, demonstrating positive market reception.
""",
            'key_findings': f"""
• Revenue Growth: {analysis['revenue_growth']:.1f}% increase over the 6-month period
• Best Performance: {analysis['best_month']} with a profit of ${analysis['best_profit']:,.0f}
• Average Monthly Revenue: ${analysis['avg_monthly_revenue']:,.0f}
• Customer Growth: Added {analysis['customer_growth']} new customers
• Total Profit Margin: {(analysis['total_profit']/analysis['total_revenue']*100):.1f}%
""",
            'recommendations': """
Based on the data analysis, we recommend the following actions:

1. Scale Marketing Efforts: The positive revenue and customer growth trends suggest strong market fit. 
   Consider increasing marketing budget by 20-30% to accelerate growth.

2. Optimize Expenses: While revenue is growing, expenses are rising proportionally. 
   Conduct a cost analysis to identify areas for efficiency improvements.

3. Customer Retention: Focus on retaining the growing customer base through loyalty programs 
   and excellent customer service.

4. Seasonal Planning: Analyze the factors that contributed to the best performing month 
   and replicate those strategies in future periods.

5. Financial Forecasting: Based on current trends, project revenue and expenses for the 
   next quarter to enable proactive decision-making.
"""
        }
        return insights
    
    def create_pdf_report(self, df, analysis, insights, chart_paths):
        """Create the final PDF report"""
        print("\n📄 Creating PDF report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"business_report_{timestamp}.pdf"
        
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Container for content
        content = []
        styles = getSampleStyleSheet()
        
        # Title
        title_text = "Business Performance Report"
        content.append(Paragraph(title_text, styles['Title']))
        content.append(Spacer(1, 0.5*inch))
        
        # Metadata
        date_text = f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
        content.append(Paragraph(date_text, styles['Normal']))
        content.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        content.append(Paragraph("Executive Summary", styles['Heading1']))
        content.append(Spacer(1, 12))
        content.append(Paragraph(insights['executive_summary'].strip(), styles['BodyText']))
        content.append(Spacer(1, 20))
        
        # Key Findings
        content.append(Paragraph("Key Findings", styles['Heading1']))
        content.append(Spacer(1, 12))
        for line in insights['key_findings'].strip().split('\n'):
            if line.strip():
                content.append(Paragraph(line.strip(), styles['BodyText']))
        content.append(Spacer(1, 20))
        
        # Data Visualizations
        content.append(Paragraph("Data Visualizations", styles['Heading1']))
        content.append(Spacer(1, 12))
        
        for chart_path in chart_paths:
            try:
                img = Image(chart_path, width=5.5*inch, height=3.3*inch)
                content.append(img)
                content.append(Spacer(1, 12))
            except Exception as e:
                print(f"  ⚠️  Could not add chart: {e}")
        
        # Recommendations
        content.append(Paragraph("Recommendations", styles['Heading1']))
        content.append(Spacer(1, 12))
        content.append(Paragraph(insights['recommendations'].strip(), styles['BodyText']))
        
        # Build PDF
        doc.build(content)
        
        print(f"  ✅ Report saved: {output_path.name}")
        return str(output_path)
    
    def generate_complete_report(self):
        """Main workflow to generate complete report"""
        print("\n" + "="*70)
        print("  AUTOMATED REPORT GENERATOR - SIMPLE VERSION")
        print("="*70)
        
        # Step 1: Create sample data
        df = self.create_sample_data()
        
        # Step 2: Analyze data
        analysis = self.analyze_data(df)
        
        # Step 3: Generate insights
        insights = self.generate_insights(analysis)
        
        # Step 4: Create charts
        chart_paths = self.create_charts(df)
        
        # Step 5: Create PDF report
        report_path = self.create_pdf_report(df, analysis, insights, chart_paths)
        
        print("\n" + "="*70)
        print("✅ REPORT GENERATION COMPLETE!")
        print("="*70)
        print(f"\n📊 Charts created: {len(chart_paths)}")
        print(f"📄 Report location: {report_path}")
        print(f"📁 Full path: {Path(report_path).absolute()}")
        print("\n" + "="*70 + "\n")
        
        return report_path


def main():
    """Main entry point"""
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     SIMPLE AUTOMATED REPORT GENERATOR                        ║
║     No API Keys Required | No Complex Setup                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    try:
        # Create generator
        generator = SimpleReportGenerator()
        
        # Generate report
        report_path = generator.generate_complete_report()
        
        # Show next steps
        print("🎉 SUCCESS! Your report is ready.")
        print("\nNext steps:")
        print("1. Open the PDF file to view your report")
        print("2. Check the 'simple_output' folder for charts")
        print("3. Modify the code to use your own data")
        print("4. Upgrade to the full version with AI and web search")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure you have installed:")
        print("  pip install pandas matplotlib reportlab")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()