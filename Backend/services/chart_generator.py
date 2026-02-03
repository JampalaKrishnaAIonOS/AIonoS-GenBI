import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)

class ChartGenerator:
    
    @staticmethod
    def detect_chart_type(question: str, df: pd.DataFrame) -> str:
        """
        Intelligently detect appropriate chart type from question and data structure.
        """
        q = (question or '').lower()
        
        # 1. Explicit user intent
        if 'pie' in q or 'donut' in q:
            return 'pie'
        if 'line' in q or 'trend' in q or 'over time' in q:
            return 'line'
        if 'barh' in q or 'horizontal' in q:
            return 'barh'
        if 'bar' in q or 'column' in q:
            return 'bar'
        if 'scatter' in q:
            return 'scatter'

        # 2. Data-driven heuristics
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = [c for c in df.columns if c not in numeric_cols]
        
        # If we have a time-like column, default to line
        time_keywords = ['date', 'time', 'year', 'month', 'day', 'hour', 'timestamp', 'period']
        if any(any(tk in col.lower() for tk in time_keywords) for col in categorical_cols):
            return 'line'
            
        # If we have many categories (e.g. > 10) and one numeric value, use horizontal bar for readability
        if len(categorical_cols) >= 1 and len(df) > 10 and len(df) < 50:
            return 'barh'
            
        # If we have a 'percentage' or 'ratio' column and few rows, pie might be good
        if any('percent' in col.lower() or 'share' in col.lower() for col in numeric_cols) and len(df) <= 10:
            return 'pie'

        return 'bar' # Default
    
    @staticmethod
    def generate_chart(data, chart_type: str = None, title: str = "Data Visualization") -> Dict[str, Any]:
        """
        Generate a comprehensive Plotly chart object from data.
        """
        try:
            if data is None or (isinstance(data, (pd.DataFrame, pd.Series)) and data.empty):
                return {"type": "error", "message": "No data available for plotting."}
            
            # 1. Standardize to DataFrame
            if isinstance(data, pd.Series):
                df = data.reset_index()
                df.columns = ['Category', 'Value']
            else:
                df = data.copy()

            # 2. Sanitize Data (JSON compatibility)
            for col in df.columns:
                if pd.api.types.is_period_dtype(df[col]) or pd.api.types.is_interval_dtype(df[col]):
                    df[col] = df[col].astype(str)
                elif not df.empty and isinstance(df[col].iloc[0], (pd.Period, pd.Interval)):
                    df[col] = df[col].astype(str)
                # Handle NaT/NaN in categorical columns
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna('N/A').astype(str)

            # 3. Detect Axis
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = [c for c in df.columns if c not in numeric_cols]

            if not numeric_cols:
                return {"type": "error", "message": "No numeric data found to plot."}

            # Map columns to X and Y
            x_col = categorical_cols[0] if categorical_cols else df.columns[0]
            y_col = numeric_cols[0] # Primary metric

            # 4. Resolve Chart Type
            if not chart_type:
                chart_type = ChartGenerator.detect_chart_type(title, df)

            # 5. Build Figure
            fig = None
            
            if chart_type == 'pie':
                fig = px.pie(df, names=x_col, values=y_col, title=title)
            elif chart_type == 'line':
                # Sort by X if it looks like a time axis
                time_keywords = ['date', 'time', 'year', 'month', 'day']
                if any(tk in x_col.lower() for tk in time_keywords):
                    df = df.sort_values(by=x_col)
                fig = px.line(df, x=x_col, y=y_col, title=title, markers=True)
            elif chart_type == 'barh':
                # Sort for better horizontal display
                df = df.sort_values(by=y_col, ascending=True)
                fig = px.bar(df, x=y_col, y=x_col, orientation='h', title=title)
            elif chart_type == 'scatter':
                color_col = categorical_cols[1] if len(categorical_cols) > 1 else None
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
            else: # bar
                df = df.sort_values(by=y_col, ascending=False)
                fig = px.bar(df, x=x_col, y=y_col, title=title)

            # 6. Apply Premium Styling
            fig.update_layout(
                template='plotly_white',
                font=dict(family="Inter, sans-serif"),
                title_font_size=20,
                hovermode="closest",
                margin=dict(l=50, r=50, t=80, b=50),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='white',
                height=500
            )

            # Improve responsiveness and aesthetics
            if chart_type != 'pie':
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', title_text=x_col)
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', title_text=y_col)

            # Special case for many labels on X axis
            if chart_type == 'bar' and len(df) > 8:
                fig.update_xaxes(tickangle=45)

            return {
                'type': 'chart',
                'chart_type': chart_type,
                'data': json.loads(fig.to_json()),
                'title': title,
                'rows_plotted': len(df),
                'columns': df.columns.tolist()
            }

        except Exception as e:
            logger.error(f"Post-processing chart failed: {e}")
            return {
                "type": "error",
                "message": f"Visualization failed: {str(e)}"
            }
