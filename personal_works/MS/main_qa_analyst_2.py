import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from pathlib import Path
from agent_qa_analyst_2 import QAAnalystAgent
import global_vars

# Initialize the agent
qa_analyst_agent = QAAnalystAgent()

def handle_agent_stream(stream_generator, chat_history):
    is_paused = False
    for chunk in stream_generator:
        if chunk.startswith("STATUS: "):
            status_message = f"🧠 **Analyst is thinking:** {chunk.replace('STATUS: ', '').strip()}"
            status_update = gr.update(visible=True, value=status_message)
            yield chat_history, status_update, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        elif chunk.startswith("WAIT_FOR_FEEDBACK"):
            is_paused = True
            plan = chunk.split("PLAN:\n", 1)[1]
            feedback_prompt = (
                "📝 **Action Required**\n\n"
                "The analyst has proposed the following plan. Please provide feedback or type 'continue' to proceed.\n\n"
                f"**Proposed Plan:**\n{plan}"
            )
            chat_history[-1][1] += f"\n\n{feedback_prompt}"
            
            yield (
                chat_history, 
                gr.update(visible=False),              # status_display
                gr.update(visible=False),              # main_input_row
                gr.update(visible=True),               # feedback_row
                gr.update(interactive=False),          # msg_textbox
                gr.update(interactive=True),           # feedback_textbox
                gr.update(interactive=True)            # feedback_btn
            )
            break

        elif chunk == "FINAL_ANSWER_START\n":
            final_status = "✅ **Investigation complete.** Generating final answer..."
            status_update = gr.update(visible=True, value=final_status)
            yield chat_history, status_update, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        
        else:
            chat_history[-1][1] += chunk
            status_update = gr.update(visible=True)
            yield chat_history, status_update, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    if not is_paused:
        yield (
            chat_history, 
            gr.update(visible=False),              # status_display
            gr.update(visible=True),               # main_input_row
            gr.update(visible=False),              # feedback_row
            gr.update(interactive=True),           # msg_textbox
            gr.update(interactive=False),          # feedback_textbox
            gr.update(interactive=False)           # feedback_btn
        )

def respond(message, chat_history):
    yield (
        chat_history, 
        gr.update(visible=True, value="*Thinking...*"), 
        gr.update(),
        gr.update(),
        gr.update(interactive=False),
        gr.update(),
        gr.update()
    )

    chat_history.append([message, ""])
    
    lc_history = []
    for user_msg, ai_msg in chat_history[:-1]:
        lc_history.append(HumanMessage(content=user_msg))
        ai_msg_clean = ai_msg.split("📝 **Action Required**")[0].strip()
        if ai_msg_clean:
            lc_history.append(AIMessage(content=ai_msg_clean))

    agent_stream = qa_analyst_agent.process_message(message, lc_history)
    yield from handle_agent_stream(agent_stream, chat_history)

### FIX 3: Appending the feedback message to the chat history ###
def handle_feedback(feedback, chat_history):
    # Append the feedback to the chat history so it's visible to the user
    # Note: We are not sending this back into the agent's history, it's just for display
    chat_history.append([f"_(Feedback)_ {feedback}", ""])

    yield (
        chat_history, 
        gr.update(visible=True, value="*Resuming with your feedback...*"),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False)
    )
    
    agent_stream = qa_analyst_agent.resume_with_feedback(feedback)
    yield from handle_agent_stream(agent_stream, chat_history)


def build_manual_ui():
    with gr.Blocks(theme=gr.themes.Glass(), title="QA Analyst Assistant") as demo:
        gr.Markdown("# QA Analyst Assistant")
        gr.Markdown("Ask questions about your MotionSolve model for root cause analysis.")

        chatbot = gr.Chatbot([], elem_id="chatbot", bubble_full_width=False, height=400)
        status_display = gr.Markdown(visible=False, elem_id="status_display")

        with gr.Row(elem_id="main_input_row") as main_input_row:
            msg_textbox = gr.Textbox(
                placeholder="e.g., Why is the 'vertical force' oscillating?",
                container=False, scale=7, elem_id="msg_textbox"
            )

        with gr.Row(visible=False, elem_id="feedback_row") as feedback_row:
            feedback_textbox = gr.Textbox(
                placeholder="Provide feedback or type 'continue' to approve the plan.",
                container=False, scale=7, elem_id="feedback_textbox"
            )
            feedback_btn = gr.Button("Submit Feedback", elem_id="feedback_btn")

        all_outputs = [
            chatbot, 
            status_display, 
            main_input_row, 
            feedback_row, 
            msg_textbox, 
            feedback_textbox, 
            feedback_btn
        ]

        msg_textbox.submit(
            respond,
            [msg_textbox, chatbot],
            all_outputs
        ).then(
            lambda: gr.update(value=""), None, [msg_textbox]
        )

        feedback_btn.click(
            handle_feedback,
            [feedback_textbox, chatbot],
            all_outputs
        ).then(
            lambda: gr.update(value=""), None, [feedback_textbox]
        )

    demo.queue()
    demo.launch()

if __name__ == "__main__":
    # Ensure global_vars.data_dir is a Path object if you use it this way
    # graph_path = global_vars.data_dir / "qa_analyst_agent_graph_2.png"
    graph_path = Path(__file__).parent / "qa_analyst_agent_graph_2.png"

    print(f"Attempting to save graph visualization to: {graph_path}")
    qa_analyst_agent.save_graph(graph_path)
    build_manual_ui()