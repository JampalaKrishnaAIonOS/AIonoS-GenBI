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
        
        if data is None or (isinstance(data, (pd.DataFrame, pd.Series)) and data.empty):
            return {
                "type": "error",
                "message": "No data available to generate visualization."
            }
        
        # Convert to appropriate format
        if isinstance(data, pd.Series):
            df = data.reset_index()
            df.columns = ['Category', 'Value']
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            return {
                "type": "error",
                "message": "Invalid data format for visualization."
            }
            
        # 🔗 Sanitize: Convert Period/Interval columns to string to avoid Plotly/JSON errors
        for col in df.columns:
            # Check for PeriodDtype or if the first element is a Period/Interval
            if pd.api.types.is_period_dtype(df[col]) or pd.api.types.is_interval_dtype(df[col]):
                df[col] = df[col].astype(str)
            elif not df[col].empty and isinstance(df[col].iloc[0], (pd.Period, pd.Interval)):
                 df[col] = df[col].astype(str)
        
        # Auto-detect chart type if not provided
        if not chart_type:
            chart_type = ChartGenerator.detect_chart_type(title, df)
        
        # Infer axis
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = [c for c in df.columns if c not in numeric_cols]

        x_col = categorical_cols[0] if categorical_cols else df.columns[0]
        y_col = numeric_cols[0] if numeric_cols else (df.columns[1] if len(df.columns) > 1 else df.columns[0])

        try:
            if chart_type == 'bar':
                fig = px.bar(df, x=x_col, y=y_col, title=title)
            elif chart_type == 'line':
                fig = px.line(df, x=x_col, y=y_col, title=title)
            elif chart_type == 'pie':
                fig = px.pie(df, names=x_col, values=y_col, title=title)
            else:
                fig = px.bar(df, x=x_col, y=y_col, title=title)
            
            # Update layout
            fig.update_layout(
                template='plotly_white',
                height=400,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            return {
                'type': 'chart',
                'chart_type': chart_type,
                'data': json.loads(fig.to_json()),
                'title': title,
                'x_axis': x_col,
                'y_axis': y_col,
                'rows_plotted': len(df)
            }
            
        except Exception as e:
            print(f"Chart generation error: {e}")
            return {
                "type": "error",
                "message": f"Failed to generate visualization: {str(e)}"
            }
