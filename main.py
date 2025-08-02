from agents.emotion_agent import EmotionDetectionAgent
from langgraph.graph import Graph
import asyncio

async def main():
    # Initialize agent
    emotion_agent = EmotionDetectionAgent()
    
    # Build workflow
    workflow = emotion_agent.build_workflow()
    
    # Compile the workflow
    app = workflow.compile()
    
    # Example usage
    texts = [
        "I'm feeling great today!",
        "This is so frustrating.",
        "I'm scared about what might happen."
    ]
    
    for text in texts:
        print(f"\nAnalyzing: '{text}'")
        result = await app.ainvoke({"text": text})
        print(f"Result: {result['response']}")

if __name__ == "__main__":
    asyncio.run(main())