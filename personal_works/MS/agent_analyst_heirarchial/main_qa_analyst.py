import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from pathlib import Path
#from agent_analyst import QAAnalystAgent
from action_step_executor_analyst import ActionStepExecutorAnalyst
import global_vars
from neo4j_kg_builder import Neo4jConnector

# Initialize the agent
neo4j_connector = Neo4jConnector(
    uri=global_vars.NEO4J_URI,
    user=global_vars.NEO4J_USER,
    password=global_vars.NEO4J_PASSWORD
    )
qa_analyst_agent = ActionStepExecutorAnalyst(neo4j_connector)

def handle_agent_stream(stream_generator, chat_history):
    is_paused = False
    for chunk in stream_generator:
        if chunk.startswith("STATUS: "):
            status_message = f"🧠 **Analyst is thinking:** {chunk.replace('STATUS: ', '').strip()}"
            status_update = gr.update(visible=True, value=status_message)
            yield chat_history, status_update, gr.update(), gr.update()
        

        elif chunk == "FINAL_ANSWER_START\n":
            final_status = "✅ **Investigation complete.** Generating final answer..."
            status_update = gr.update(visible=True, value=final_status)
            yield chat_history, status_update, gr.update(), gr.update()
        
        else:
            chat_history[-1][1] += chunk
            status_update = gr.update(visible=True)
            yield chat_history, status_update, gr.update(), gr.update()

    if not is_paused:
        yield (
            chat_history, 
            gr.update(visible=True),              # status_display
            gr.update(visible=True),               # main_input_row
            gr.update(visible=True),              # feedback_row            
        )

def respond(message, chat_history):
    yield (
        chat_history, 
        gr.update(visible=True, value="*Thinking...*"), 
        gr.update(),
        gr.update(),
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



def build_manual_ui():
    with gr.Blocks(theme=gr.themes.Glass(), title="Model Analyst Assistant") as demo:
        gr.Markdown("# Model Analyst Assistant")
        gr.Markdown("Initiate a discussion about your MotionSolve model for root cause analysis.")

        chatbot = gr.Chatbot([], elem_id="chatbot", bubble_full_width=False, height=400)
        status_display = gr.Markdown(visible=False, elem_id="status_display")

        with gr.Row(elem_id="main_input_row") as main_input_row:
            msg_textbox = gr.Textbox(
                placeholder="e.g., Why is the 'vertical force' oscillating?",
                container=False, scale=7, elem_id="msg_textbox"
            )

        

        all_outputs = [
            chatbot, 
            status_display, 
            main_input_row, 
            msg_textbox, 
        ]

        msg_textbox.submit(
            respond,
            [msg_textbox, chatbot],
            all_outputs
        ).then(
            lambda: gr.update(value=""), None, [msg_textbox]
        )


    demo.queue()
    demo.launch()

if __name__ == "__main__":
    # Ensure global_vars.data_dir is a Path object if you use it this way
    # graph_path = global_vars.data_dir / "qa_analyst_agent_graph_2.png"
    # graph_path = Path(__file__).parent / "qa_analyst_agent_graph_2.png"

    # print(f"Attempting to save graph visualization to: {graph_path}")
    # qa_analyst_agent.save_graph(graph_path)
    build_manual_ui()

