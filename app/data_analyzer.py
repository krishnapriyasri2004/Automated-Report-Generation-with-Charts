import pandas as pd
import numpy as np
from typing import Dict, List, Any

class DataAnalyzer:
    """Analyzes data and identifies patterns for visualization"""
    
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a pandas DataFrame and suggest visualizations
        """
        print(f"📊 Analyzing data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'missing_values': df.isnull().sum().to_dict(),
            'summary_stats': df.describe().to_dict() if not df.empty else {},
            'suggested_charts': []
        }
        
        # Suggest chart types based on data
        if len(analysis['numeric_columns']) >= 1:
            analysis['suggested_charts'].append({
                'type': 'histogram',
                'columns': analysis['numeric_columns'][:3],
                'description': 'Distribution of numeric values'
            })
        
        if len(analysis['numeric_columns']) >= 2:
            analysis['suggested_charts'].append({
                'type': 'scatter',
                'x': analysis['numeric_columns'][0],
                'y': analysis['numeric_columns'][1],
                'description': 'Relationship between two variables'
            })
        
        if len(analysis['categorical_columns']) >= 1 and len(analysis['numeric_columns']) >= 1:
            analysis['suggested_charts'].append({
                'type': 'bar',
                'category': analysis['categorical_columns'][0],
                'value': analysis['numeric_columns'][0],
                'description': 'Comparison across categories'
            })
        
        return analysis
    
    def find_trends(self, df: pd.DataFrame, date_column: str = None) -> Dict:
        """
        Identify trends in time-series data
        """
        if date_column and date_column in df.columns:
            df_sorted = df.sort_values(by=date_column)
            numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns
            
            trends = {}
            for col in numeric_cols:
                values = df_sorted[col].dropna()
                if len(values) > 1:
                    # Simple trend calculation
                    trend = "increasing" if values.iloc[-1] > values.iloc[0] else "decreasing"
                    percent_change = ((values.iloc[-1] - values.iloc[0]) / values.iloc[0]) * 100
                    trends[col] = {
                        'trend': trend,
                        'change_percent': round(percent_change, 2)
                    }
            
            return trends
        
        return {}
