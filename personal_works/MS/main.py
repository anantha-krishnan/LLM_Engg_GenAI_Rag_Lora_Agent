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
    for chunk in mdl_agent.process_message(message, lc_history):
        bot_response += chunk
        #chat_history[-1][1] = bot_response
        #yield chat_history  # Stream the updated chat history
        yield bot_response

def add_user_message(message, chat_history):
    """Adds the user's message to the chat history."""
    if message.strip() == "":
        return chat_history
    return chat_history + [[message, None]]
     

def build_manual_ui():
    """Builds and launches the Gradio chat interface manually."""
    with gr.Blocks(theme=gr.themes.Glass(), title="MDL Modeling Assistant") as demo:
        gr.Markdown("# Vehicle Dynamics Modeling Assistant")
        gr.Markdown("Ask questions to find components in the MDL library.")
        
        chatbot = gr.Chatbot([],
            elem_id="chatbot",
            bubble_full_width=False,
            height=600,
            avatar_images=(None, "https://i.imgur.com/u5t7f2L.png")
        )
        with gr.Row():
            msg = gr.Textbox(placeholder="e.g., What types of suspensions are available?", container=False, scale=7)
        with gr.Row():            
            clear_button = gr.Button("Clear Conversation & Start New Session")
        
        msg.submit(add_user_message, [msg, chatbot], [chatbot], queue=False).then(
            respond, [msg, chatbot], [chatbot]
        )
        clear_button.click(lambda: None, None, chatbot, queue=False)
    demo.launch()

def build_ui():
    """Builds and launches the Gradio chat interface."""
    with gr.Blocks(theme=gr.themes.Glass(), title="MDL Modeling Assistant") as demo:
        gr.Markdown("# Vehicle Dynamics Modeling Assistant")
        gr.Markdown("Ask questions to find components in the MDL library.")
        
        chatbot = gr.ChatInterface(
            respond,
            chatbot=gr.Chatbot(height=400),
            textbox=gr.Textbox(placeholder="e.g., What types of suspensions are available?", container=False, scale=7),
            title=None,
            examples=[
                "List all available suspensions",
                "I need a rear suspension for a truck",
                "Find a rack and pinion steering system",
                "Show me all the example models"
            ],
        )
        with gr.Row():
            clear_button = gr.Button("Clear Conversation & Start New Session")
        clear_button.click(None, None, [chatbot], queue=False)
    demo.launch()

if __name__ == "__main__":
    build_ui()