# main.py
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from agent import mdl_agent # Import the agent instance

def respond(message: str, chat_history: list):
    """
    Gradio response function. It converts Gradio's history format to LangChain's,
    invokes the agent, and streams the response back.
    """
    # Convert Gradio history to LangChain message format
    lc_history = []
    for user_msg, ai_msg in chat_history:
        lc_history.append(HumanMessage(content=user_msg))
        lc_history.append(AIMessage(content=ai_msg))

    bot_response = ""
    # The agent's invoke method now returns a stream
    for chunk in mdl_agent.invoke(message, lc_history):
        bot_response += chunk
        yield bot_response

def build_ui():
    """Builds and launches the Gradio chat interface."""
    with gr.Blocks(theme=gr.themes.Soft(), title="MDL Modeling Assistant") as demo:
        gr.Markdown("# Vehicle Dynamics Modeling Assistant")
        gr.Markdown("Ask questions to find components in the MDL library.")
        
        chatbot = gr.ChatInterface(
            respond,
            chatbot=gr.Chatbot(height=600),
            textbox=gr.Textbox(placeholder="e.g., What types of suspensions are available?", container=False, scale=7),
            title=None,
            examples=[
                "List all available suspensions",
                "I need a rear suspension for a truck",
                "Find a rack and pinion steering system",
                "Show me all the example models"
            ],
        )
    demo.launch()

if __name__ == "__main__":
    build_ui()