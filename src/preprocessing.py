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

def preprocessing_node(state: FinancialState) -> FinancialState:
    if state.error or not state.raw_data:
        return state
    
    state_dict = state.model_dump()
    
    try:
        # Check if raw_data contains stock information
        if "stock" not in state.raw_data:
            raise ValueError("No stock data found in raw_data")
        
        stock_data = state.raw_data["stock"]
        
        # Handle different API response formats
        time_series_key = None
        for possible_key in ["Time Series (Daily)", "Time Series (5min)", "Time Series"]:
            if possible_key in stock_data:
                time_series_key = possible_key
                break
        
        if not time_series_key:
            raise ValueError("Could not find time series data in stock response")
        
        # Process stock data with proper date sorting
        stock_df = pd.DataFrame(stock_data[time_series_key]).T.rename(columns={
            "1. open": "Open",
            "4. close": "Close"
        })
        stock_df.index = pd.to_datetime(stock_df.index)
        stock_df = stock_df.sort_index(ascending=True)  # Ensure ascending order
        stock_df = stock_df[["Open", "Close"]].astype(float)
        
        # Process news with Unicode handling
        news_content = ""
        if "news" in state.raw_data and "articles" in state.raw_data["news"]:
            news_content = "\n".join([
                f"{article['title']}: {article['description']}"
                for article in state.raw_data["news"]["articles"]
                if article.get('description')
            ]).encode('utf-8', 'ignore').decode('utf-8')[:5000]  # Limit context length
        
        state_dict["processed_data"] = {
            "stock": stock_df,
            "news": news_content
        }
        state_dict["error"] = None
    
    except Exception as e:
        state_dict["error"] = f"Preprocessing failed: {str(e)}"
    
    return FinancialState(**state_dict)