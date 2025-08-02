from langgraph.graph import Graph
from typing import Dict, Any
from transformers import pipeline
import requests
import os
from dotenv import load_dotenv

load_dotenv()

class EmotionDetectionAgent:
    def __init__(self):
        # Initialize free emotion detection model from HuggingFace
        self.local_model = pipeline(
            "text-classification", 
            model="finiteautomata/bertweet-base-emotion-analysis"
        )
        
        # Alternatively, you can use free APIs
        self.use_api = False
        self.api_url = "https://api-inference.huggingface.co/models/finiteautomata/bertweet-base-emotion-analysis"
        self.api_key = os.getenv("HF_API_KEY", "")
    
    def detect_emotion_local(self, text: str) -> Dict[str, Any]:
        """Detect emotion using local model"""
        result = self.local_model(text)
        return {
            "emotion": result[0]['label'],
            "confidence": result[0]['score'],
            "model": "local"
        }
    
    def detect_emotion_api(self, text: str) -> Dict[str, Any]:
        """Detect emotion using free API"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"inputs": text}
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list):
                return {
                    "emotion": result[0][0]['label'],
                    "confidence": result[0][0]['score'],
                    "model": "api"
                }
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def detect_emotion(self, text: str) -> Dict[str, Any]:
        """Detect emotion with fallback logic"""
        if self.use_api and self.api_key:
            result = self.detect_emotion_api(text)
            if 'error' not in result:
                return result
        
        # Fallback to local model
        return self.detect_emotion_local(text)
    
    def build_workflow(self) -> Graph:
        """Build LangGraph workflow for emotion detection"""
        workflow = Graph()
        
        workflow.add_node("detect_emotion", self.detect_emotion)
        workflow.add_node("format_response", lambda x: {"response": x})
        
        workflow.add_edge("detect_emotion", "format_response")
        
        workflow.set_entry_point("detect_emotion")
        workflow.set_finish_point("format_response")
        
        return workflow