import sys
import io
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pmdarima import auto_arima
from langgraph.graph import StateGraph, END
from sklearn.ensemble import IsolationForest
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime, timedelta
import os
import re
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Define enhanced state model
class FinancialState(BaseModel):
    user_query: str = Field(..., description="Original user input")
    symbol: Optional[str] = Field(None, description="Detected stock symbol")
    raw_data: Optional[Dict[str, Any]] = None
    processed_data: Optional[Dict[str, Any]] = None
    model: Optional[Any] = None
    predictions: Optional[List[float]] = None
    anomalies: Optional[List[int]] = None
    insights: Optional[str] = None
    visualizations: List[str] = []
    error: Optional[str] = None

def report_node(state: FinancialState) -> FinancialState:
    """Generate final report"""
    if state.error:
        print(f"⚠️ Skipping report due to error: {state.error}")
        return state
    
    if not state.symbol:
        error_msg = "Cannot generate report - missing symbol"
        print(f"⚠️ {error_msg}")
        return FinancialState(**state.model_dump(), error=error_msg)
    
    state_dict = state.model_dump()
    # filename = state_dict["report_link"]
    print(f"\n=== ENTERING REPORT NODE ===")  # Debug
    
    
    filename = f"reports/{state.symbol}_report.html"

    try:
        report = f"""
        <html>
            <head>
                <meta charset="UTF-8">
                <title>{state.symbol} Analysis Report</title>
                <style>
                    body {{ font-family: Arial Unicode MS, sans-serif; margin: 2rem; }}
                    .insights {{ white-space: pre-wrap; line-height: 1.6; }}
                    .visualization {{ margin: 2rem 0; border: 1px solid #ddd; padding: 1rem; }}
                </style>
            </head>
            <body>
                <h1>{state.symbol} Financial Analysis</h1>
                <div class="insights">{state.insights}</div>
                {"".join(f'<div class="visualization">{viz}</div>' for viz in state.visualizations)}
            </body>
        </html>
        """
        

        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"✅ Report generated at: {filename}")
        
       
        return FinancialState(**state_dict)
    
    except Exception as e:
        state_dict["error"] = f"Report generation failed: {str(e)}"
    

    return FinancialState(**state_dict)