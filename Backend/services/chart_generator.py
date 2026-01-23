import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional
import json

class ChartGenerator:
    
    @staticmethod
    def detect_chart_type(question: str, data) -> str:
        """Detect appropriate chart type from question.

        Priority order:
        1. Explicit chart keywords (pie/line/bar) in user's text
        2. Semantic hints (trend/distribution/compare)
        3. Safe default 'bar'
        """
        q = (question or '').lower()

        # explicit user intent (highest priority)
        if 'pie' in q:
            return 'pie'
        if 'line' in q:
            return 'line'
        if 'bar' in q:
            return 'bar'

        # semantic hints
        if any(k in q for k in ['trend', 'over time', 'timeline', 'monthly', 'yearly']):
            return 'line'
        if any(k in q for k in ['distribution', 'share', 'percentage', 'portion']):
            return 'pie'
        if any(k in q for k in ['compare', 'comparison', 'vs']):
            return 'bar'

        # safe default
        return 'bar'
    
    @staticmethod
    def generate_chart(data, chart_type: str = None, title: str = "Analysis Result") -> Dict[str, Any]:
        """Generate Plotly chart from data"""
        
        if data is None:
            return None
        
        # Convert to appropriate format
        if isinstance(data, pd.Series):
            df = data.reset_index()
            df.columns = ['Category', 'Value']
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            return None
        
        # Auto-detect chart type if not provided
        if not chart_type:
            chart_type = ChartGenerator.detect_chart_type(title, df)
        
        try:
            if chart_type == 'bar':
                fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=title)
            elif chart_type == 'line':
                fig = px.line(df, x=df.columns[0], y=df.columns[1], title=title)
            elif chart_type == 'pie':
                fig = px.pie(df, names=df.columns[0], values=df.columns[1], title=title)
            else:
                fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=title)
            
            # Update layout
            fig.update_layout(
                template='plotly_white',
                height=400,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            return {
                'chart_type': chart_type,
                'data': json.loads(fig.to_json()),
                'title': title
            }
            
        except Exception as e:
            print(f"Chart generation error: {e}")
            return None