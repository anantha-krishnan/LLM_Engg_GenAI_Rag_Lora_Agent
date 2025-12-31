import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
import plotly.graph_objects as go
#from agent_analyst import QAAnalystAgent
from action_step_executor_analyst import ActionStepExecutorAnalyst
import global_vars
from neo4j_kg_builder import Neo4jConnector
import base64
import json

head_js = """
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
<script>
    mermaid.initialize({ startOnLoad: false });
</script>
"""

# Initialize the agent
neo4j_connector = Neo4jConnector(
    uri=global_vars.NEO4J_URI,
    user=global_vars.NEO4J_USER,
    password=global_vars.NEO4J_PASSWORD
    )
qa_analyst_agent = ActionStepExecutorAnalyst(neo4j_connector)


def wrap_mermaid_html(mermaid_code: str) -> str:
    """
    Renders Mermaid graphs using v9 (Synchronous) inside an IFrame.
    Confirmed working with UNPKG for Pan/Zoom.
    """
    if not mermaid_code:
        return "<div style='padding:20px; text-align:center; color:gray;'><i>No topology identified yet.</i></div>"

    # Safely encode the mermaid string for JS (handles quotes/newlines)
    js_safe_code = json.dumps(mermaid_code)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: sans-serif; margin: 0; padding: 0; background-color: #ffffff; }}
            
            /* Container: Allows standard scrolling if PanZoom fails */
            #graph-container {{ 
                width: 100vw; 
                height: 100vh; 
                overflow: hidden; /* PanZoom handles movement */
                display: flex; 
                justify-content: center; 
                align-items: center; 
            }}
            
            #status {{ 
                position: absolute; top: 5px; left: 5px; 
                background: rgba(255,255,255,0.9); 
                padding: 4px 8px; font-size: 12px; color: #555; 
                border-radius: 4px; pointer-events: none;
                z-index: 1000;
            }}
            
            svg {{ 
                max-width: none !important; 
            }}
        </style>
        
        <!-- 1. MERMAID v9 (Synchronous) -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/mermaid/9.4.3/mermaid.min.js"></script>
        
        <!-- 2. PANZOOM (UNPKG) -->
        <script src="https://unpkg.com/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
    </head>
    <body>
        <div id="status">Loading...</div>
        <div id="graph-container"></div>

        <script>
            var container = document.getElementById('graph-container');
            var statusDiv = document.getElementById('status');
            var graphDefinition = {js_safe_code};

            // 1. Check Mermaid
            if (typeof mermaid === 'undefined') {{
                statusDiv.innerText = "❌ Network Error: Mermaid failed to load.";
                statusDiv.style.color = "red";
            }} else {{
                // 2. Initialize
                mermaid.initialize({{
                    startOnLoad: false,
                    securityLevel: 'loose',
                    flowchart: {{ useMaxWidth: false, htmlLabels: true }}
                }});

                // 3. Render
                try {{
                    mermaid.render('main-svg-graph', graphDefinition, function(svgCode) {{
                        container.innerHTML = svgCode;
                        var svgElement = container.querySelector('svg');
                        
                        // Ensure SVG fills space for PanZoom
                        svgElement.setAttribute('width', '100%');
                        svgElement.setAttribute('height', '100%');

                        // 4. Attach PanZoom
                        if (typeof svgPanZoom !== 'undefined') {{
                            svgPanZoom(svgElement, {{
                                zoomEnabled: true,
                                controlIconsEnabled: true,
                                fit: true,
                                center: true,
                                minZoom: 0.1,
                                maxZoom: 20
                            }});
                            // Success! Hide status message after 1 second
                            statusDiv.innerText = "✅ Interactive";
                            statusDiv.style.color = "green";
                            setTimeout(function(){{ statusDiv.style.display = 'none'; }}, 1000);
                        }} else {{
                            // Fallback if PanZoom gets blocked later
                            statusDiv.innerText = "⚠️ Zoom blocked - Use Scrollbars";
                            container.style.overflow = 'auto';
                            svgElement.style.width = '150%';
                        }}
                    }});
                }} catch (err) {{
                    statusDiv.innerText = "Syntax Error";
                    container.innerHTML = "<pre style='color:red; padding:20px;'>" + err.message + "</pre>";
                }}
            }}
        </script>
    </body>
    </html>
    """

    # Encode to Base64 to bypass Gradio/Browser HTML injection restrictions
    b64_html = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')

    # Return IFrame with adjustable height (75vh = 75% of view height)
    return f"""
    <iframe 
        src="data:text/html;base64,{b64_html}" 
        style="width: 100%; height: 75vh; border: 1px solid #e5e7eb; border-radius: 8px; background: white;"
        frameborder="0">
    </iframe>
    """
def handle_agent_stream(stream_generator, chat_history):
    # Default values for all UI components
    status_msg = ""
    plot_data = gr.update()
    chain_data = gr.update()
    plan_data = gr.update()

    for chunk in stream_generator:
        msg_type = chunk.get("type")
        data = chunk.get("data")

        if msg_type == "status":
            status_msg = data
            yield chat_history, status_msg, plot_data, chain_data, plan_data

        elif msg_type == "plan":
            # Format the plan as a Markdown checklist
            plan_markdown = "### 📋 Current Plan\n" + "\n".join([f"- {s}" for s in data])
            plan_data = gr.update(value=plan_markdown)
            yield chat_history, status_msg, plot_data, chain_data,  plan_data

        elif msg_type == "chain":
            # data should be the mermaid code string
            rendered_html = wrap_mermaid_html(data)
            topology_html = gr.update(value=rendered_html)
            yield chat_history, status_msg, plot_data, topology_html,  plan_data

        elif msg_type == "plot":
            try:
                if isinstance(data, dict):
                    fig_obj = go.Figure(data)
                    plot_data = gr.update(value=fig_obj)
                # elif data is not None:
                #     plot_data = gr.update(value=data)
            except Exception as e:
                print(f"Error reconstructing plot: {e}")
                plot_data = gr.update()
            yield chat_history, status_msg, plot_data, chain_data,  plan_data

        elif msg_type == "text":
            # Accumulate final text in chat
            if not chat_history[-1][1]:
                chat_history[-1][1] = "✅ **Analysis Result:**\n"
            chat_history[-1][1] += data
            yield chat_history, "✅ Finished", plot_data, chain_data,  plan_data

    # Final yield to show everything is done
    yield chat_history, "✅ Investigation complete.", plot_data, chain_data,  plan_data
    
def respond(message, chat_history):
    yield (
        chat_history, 
        gr.update(visible=True, value="*Thinking...*"), 
        gr.update(),
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
    with gr.Blocks(head=head_js, theme=gr.themes.Glass(), title="MBD Model Analyst") as demo:
        gr.Markdown("# 🛠️ MotionSolve Model Analyst")
        
        with gr.Row():
            # LEFT COLUMN: Communication
            with gr.Column(scale=1.5):
                chatbot = gr.Chatbot([], elem_id="chatbot", height=500)
                status_display = gr.Markdown("Ready", elem_id="status_display")
                
                with gr.Row():
                    msg_textbox = gr.Textbox(
                        placeholder="Ask about forces, joints, or oscillations...",
                        container=False, scale=7
                    )
                
                plan_display = gr.Markdown("### 📋 Current Plan\n*No active plan*")

            # RIGHT COLUMN: Evidence & Data
            with gr.Column(scale=1):
                with gr.Tabs():
                    
                    with gr.TabItem("🕸️ Model Topology"):
                        # Show the Condensed Chain here
                        model_chain_view = gr.HTML(label="Causal Chain Diagram")
                        gr.Markdown("*Mermaid Knowledge graph will appear here...*")

                    with gr.TabItem("📊 Data Analysis"):
                        # This will hold the Plotly charts from python_analysis
                        analysis_plot = gr.Plot(label="Simulation Data")
                        analysis_summary = gr.Markdown("*Numerical findings will appear here...*")


        # Update handle_agent_stream to update these new components
        # (You will need to modify your stream generator to yield structured data)
        msg_textbox.submit(
            respond,
            [msg_textbox, chatbot],
            [chatbot, status_display, analysis_plot, model_chain_view, plan_display]
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

