# main_qa_analyst.py
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from pathlib import Path
from agent_qa_test_analyst import QAAnalystAgent  # Import the QA analyst agent instance
#from crew_analyst_agent import CrewAIAnalystAgent
import global_vars

qa_analyst_agent = QAAnalystAgent()

def respond(message, chat_history):
    """
    Gradio response function. It converts Gradio's history format to LangChain's,
    invokes the QA analyst agent, and streams the response back.
    """
    chat_history.append([message, ""])  # Append user message with empty AI response

    # Convert Gradio history to LangChain message format
    lc_history = []
    for user_msg, ai_msg in chat_history[:-1]:  # Exclude the last message which is being processed
        lc_history.append(HumanMessage(content=user_msg))
        lc_history.append(AIMessage(content=ai_msg))

    # The agent's invoke method now returns a stream
    for chunk in qa_analyst_agent.process_message(message, lc_history):
        chat_history[-1][1] += chunk  # Update the last AI response: for langraph agent
        #chat_history[-1][1] = chunk  # Update the last AI response: for crewai agent
        yield chat_history  # Stream the updated chat history

def add_user_message(message, chat_history):
    """Adds the user's message to the chat history."""
    if message.strip() == "":
        return chat_history
    return chat_history + [[message, None]]

def build_manual_ui():
    """Builds and launches the Gradio chat interface manually."""
    with gr.Blocks(theme=gr.themes.Glass(), title="QA Analyst Assistant") as demo:
        gr.Markdown("# QA Analyst Assistant")
        gr.Markdown("Ask questions related to quality assurance analysis.")
        
        chatbot = gr.Chatbot([],
            elem_id="chatbot",
            height=600,
        )
        with gr.Row():
            msg = gr.Textbox(placeholder="e.g., What are the full car test models available?", container=False, scale=7)
        def clear_msg():
            return ""
        msg.submit(respond, [msg, chatbot], [chatbot]).then(
            clear_msg, None, [msg]
        )
    demo.queue()
    demo.launch()
if __name__ == "__main__":
    qa_analyst_agent.save_graph(global_vars.data_dir / "model_analyst_langgraph.png")
    build_manual_ui()

# the output 'Body 1-left(Output 0)' reduces gradually. why is it so?    
# is my "vertical force" from "AutoTire - Tire CP Forces (W-Axis system)" not converging? 
# can you explain the Force type entities in the model