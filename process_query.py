import sys
import io
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from src.preprocessing import preprocessing_node
from src.data_ingestion import data_ingestion_node
from src.model_training import model_training_node
from src.prediction import prediction_node
from src.anomaly_detection import anomaly_detection_node
from src.insight_generation import insight_generation_node
from src.visualization import visualization_node
from src.report import report_node
from src.process_query import process_query_node

# Fix Unicode handling system-wide
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

load_dotenv()

# Initialize Groq client
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.1,
    max_tokens=1024,
    api_key=os.environ["GROQ_API_KEY"]
)

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
    report_link: str = ""
    error: Optional[str] = None

# Define workflow
workflow = StateGraph(FinancialState)

nodes = [
    process_query_node,
    data_ingestion_node,
    preprocessing_node,
    model_training_node,
    prediction_node,
    anomaly_detection_node,
    insight_generation_node,
    visualization_node,
    report_node
]

for node in nodes:
    workflow.add_node(node.__name__, node)

workflow.set_entry_point("process_query_node")
workflow.add_edge("process_query_node", "data_ingestion_node")
workflow.add_edge("data_ingestion_node", "preprocessing_node")
workflow.add_edge("preprocessing_node", "model_training_node")
workflow.add_edge("model_training_node", "prediction_node")
workflow.add_edge("prediction_node", "anomaly_detection_node")
workflow.add_edge("anomaly_detection_node", "insight_generation_node")
workflow.add_edge("insight_generation_node", "visualization_node")
workflow.add_edge("visualization_node", "report_node")
workflow.add_edge("report_node", END)

chain = workflow.compile()
