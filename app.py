"""
Chainlit Frontend for Text2SQL Chatbot with Graph Visualization and LangGraph Debugging
"""

import chainlit as cl
from text2sql_agent import process_question_stream, generate_graph_visualization
import json

##Generate workflow diagram once at module load (optional)
##Uncomment the lines below if you want to generate the diagram:
try:
    workflow_diagram_path = generate_graph_visualization("text2sql_workflow.png")
    if workflow_diagram_path:
        print(f"✅ Workflow diagram generated: {workflow_diagram_path}")
except Exception as e:
    print(f"⚠️ Warning: Could not generate workflow diagram: {e}")

# Set page configuration
@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    
    await cl.Message(
        content="👋 Welcome to the Text2SQL E-commerce Assistant!\n\n"
                "I can help you query the e-commerce database using natural language. "
                "Just ask me questions about:\n"
                "- Orders and their status\n"
                "- Customers and their locations\n"
                "- Products and categories\n"
                "- Payments and transactions\n"
                "- Reviews and ratings\n"
                "- Sellers and their information\n\n"
                "**Example questions:**\n"
                "- How many orders were delivered?\n"
                "- What are the top 5 product categories by sales?\n"
                "- Show me orders from São Paulo\n"
                "- What's the average review score?\n"
                "- Which sellers have the most orders?\n\n"
                "Go ahead and ask me anything! 🚀"
    ).send()