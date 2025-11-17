# main_qa_analyst.py
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from pathlib import Path
from agent_qa_analyst_2 import QAAnalystAgent
import global_vars

# Initialize the agent
qa_analyst_agent = QAAnalystAgent()

def respond(message, chat_history):
    """
    Gradio response function. It converts Gradio's history format to LangChain's,
    invokes the QA analyst agent, and streams the response back, handling both
    status updates and the final answer.
    """
    # --- CORRECTED SYNTAX: ---
    # To update a component, simply yield the new value/properties for it.
    # We create a dictionary for the status_display to set its 'value' and 'visible' properties.
    initial_status_update = gr.update(visible=True, value="*Thinking...*")
    yield chat_history, initial_status_update

    # Append user message with an empty placeholder for the AI's response
    chat_history.append([message, ""])

    # Convert Gradio history to LangChain message format
    lc_history = []
    for user_msg, ai_msg in chat_history[:-1]:
        lc_history.append(HumanMessage(content=user_msg))
        lc_history.append(AIMessage(content=ai_msg))

    # The agent's process_message method now returns a stream of different message types
    for chunk in qa_analyst_agent.process_message(message, lc_history):
        # --- CORRECTED SYNTAX in the loop ---

        if chunk.startswith("STATUS: "):
            # It's a reasoning log update. Update the status_display component.
            status_message = f"🧠 **Analyst is thinking:** {chunk.replace('STATUS: ', '').strip()}"
            status_update = gr.update(visible=True, value=status_message)
            yield chat_history, status_update

        elif chunk == "FINAL_ANSWER_START\n":
            # The final answer is about to start. Update the status one last time.
            final_status = "✅ **Investigation complete.** Generating final answer..."
            status_update = gr.update(visible=True, value=final_status)
            yield chat_history, status_update

        else:
            # It's a part of the final answer. Append it to the chatbot history.
            chat_history[-1][1] += chunk
            # Keep the status visible while the answer streams
            status_update = gr.update(visible=True)
            yield chat_history, status_update

    # --- CORRECTED SYNTAX: Hide the status display once the final answer is complete ---
    final_status_update = gr.update(visible=False)
    yield chat_history, final_status_update

def build_manual_ui():
    """Builds and launches the Gradio chat interface with a status display."""
    with gr.Blocks(theme=gr.themes.Glass(), title="QA Analyst Assistant") as demo:
        gr.Markdown("# QA Analyst Assistant")
        gr.Markdown("Ask questions about your MotionSolve model for root cause analysis.")

        chatbot = gr.Chatbot([],
            elem_id="chatbot",
            bubble_full_width=False,
            height=400,
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="e.g., Why is the 'vertical force' from 'AutoTire - Tire CP Forces (W-Axis system)' oscillating?",
                container=False,
                scale=7
            )
        status_display = gr.Markdown(visible=False, elem_id="status_display")

        def clear_msg():
            return ""

        msg.submit(
            respond,
            [msg, chatbot],
            [chatbot, status_display]
        ).then(
            clear_msg, None, [msg]
        )

    demo.queue()
    demo.launch()

if __name__ == "__main__":
    graph_path = global_vars.data_dir / "qa_analyst_agent_graph_2.png"
    print(f"Attempting to save graph visualization to: {graph_path}")
    qa_analyst_agent.save_graph(graph_path)
    build_manual_ui()

# the output 'Body 1-left(Output 0)' reduces gradually. why is it so?    
# is my "vertical force" from "AutoTire - Tire CP Forces (W-Axis system)" not converging? 
# can you explain the Force type entities in the model