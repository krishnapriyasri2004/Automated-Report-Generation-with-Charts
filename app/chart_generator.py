import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, Any

class ChartGenerator:
    """Generates various types of charts from data"""
    
    def __init__(self, output_dir: str = "output/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
    
    def generate_chart(self, data: pd.DataFrame, chart_config: Dict[str, Any]) -> str:
        """
        Generate a chart based on configuration
        Returns path to saved chart image
        """
        chart_type = chart_config.get('type', 'bar')
        
        if chart_type == 'bar':
            return self._create_bar_chart(data, chart_config)
        elif chart_type == 'line':
            return self._create_line_chart(data, chart_config)
        elif chart_type == 'scatter':
            return self._create_scatter_chart(data, chart_config)
        elif chart_type == 'histogram':
            return self._create_histogram(data, chart_config)
        elif chart_type == 'pie':
            return self._create_pie_chart(data, chart_config)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
    
    def _create_bar_chart(self, data: pd.DataFrame, config: Dict) -> str:
        """Create a bar chart"""
        plt.figure(figsize=(10, 6))
        
        if 'category' in config and 'value' in config:
            x = data[config['category']]
            y = data[config['value']]
            plt.bar(x, y, color='steelblue')
            plt.xlabel(config['category'])
            plt.ylabel(config['value'])
        else:
            data.plot(kind='bar')
        
        plt.title(config.get('title', 'Bar Chart'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = self.output_dir / f"bar_chart_{config.get('title', 'chart').replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Created bar chart: {filename}")
        return str(filename)
    
    def _create_line_chart(self, data: pd.DataFrame, config: Dict) -> str:
        """Create a line chart"""
        plt.figure(figsize=(10, 6))
        
        if 'x' in config and 'y' in config:
            plt.plot(data[config['x']], data[config['y']], marker='o', linewidth=2)
            plt.xlabel(config['x'])
            plt.ylabel(config['y'])
        else:
            data.plot(kind='line', marker='o')
        
        plt.title(config.get('title', 'Line Chart'))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = self.output_dir / f"line_chart_{config.get('title', 'chart').replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Created line chart: {filename}")
        return str(filename)
    
    def _create_scatter_chart(self, data: pd.DataFrame, config: Dict) -> str:
        """Create a scatter plot"""
        plt.figure(figsize=(10, 6))
        
        if 'x' in config and 'y' in config:
            plt.scatter(data[config['x']], data[config['y']], alpha=0.6, s=50)
            plt.xlabel(config['x'])
            plt.ylabel(config['y'])
        
        plt.title(config.get('title', 'Scatter Plot'))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = self.output_dir / f"scatter_{config.get('title', 'chart').replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🔵 Created scatter plot: {filename}")
        return str(filename)
    
    def _create_histogram(self, data: pd.DataFrame, config: Dict) -> str:
        """Create a histogram"""
        plt.figure(figsize=(10, 6))
        
        if 'columns' in config:
            for col in config['columns']:
                if col in data.columns:
                    plt.hist(data[col].dropna(), bins=30, alpha=0.5, label=col)
            plt.legend()
        else:
            data.hist(bins=30, figsize=(10, 6))
        
        plt.title(config.get('title', 'Distribution'))
        plt.tight_layout()
        
        filename = self.output_dir / f"histogram_{config.get('title', 'chart').replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Created histogram: {filename}")
        return str(filename)
    
    def _create_pie_chart(self, data: pd.DataFrame, config: Dict) -> str:
        """Create a pie chart"""
        plt.figure(figsize=(8, 8))
        
        if 'values' in config and 'labels' in config:
            plt.pie(data[config['values']], labels=data[config['labels']], autopct='%1.1f%%')
        
        plt.title(config.get('title', 'Pie Chart'))
        plt.tight_layout()
        
        filename = self.output_dir / f"pie_{config.get('title', 'chart').replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🥧 Created pie chart: {filename}")
        return str(filename)
