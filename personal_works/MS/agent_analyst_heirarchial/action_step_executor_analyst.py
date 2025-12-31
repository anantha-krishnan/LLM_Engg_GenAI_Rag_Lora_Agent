import operator, re
from typing import List, TypedDict, Annotated, Dict, Optional,Generator
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
import json
from typing import Annotated, List, Union, Optional
import base64
from agent_tools_2 import ToolBelt
from pathlib import Path
import global_vars

def reduce_past_steps(current: List[str], update: Union[List[str], str, None]) -> List[str]:
    # 1. If we receive None, we reset to a blank list
    if update is None:
        return []
    
    # 2. If the update is a string (single step), make it a list
    if isinstance(update, str):
        update = [update]
    
    # 3. Standard Append logic (same as operator.add)
    return current + update

# --- STATE DEFINITION ---
class ExecutionState(TypedDict):
    goal: str
    hypothesis: str
    plan: List[str]
    past_steps: Annotated[List[str], reduce_past_steps]
    #discovered_entities: Dict[str, str] # CRITICAL: Stores {"hub": "Hub_Body_01"}
    iteration_count: int
    final_answer: str
    next_step: Optional[str]
    next_action: str # to continue or replan
    last_plot: Optional[str]
    schema: str

    
class ActionStepExecutorAnalyst:
    def __init__(self, connector):
        self.llm = ChatOpenAI(model=global_vars.model_openai_4omini, temperature=0)
        self.neo4j_connector = connector
        self.tool_belt = ToolBelt(connector,self.llm)
        self.subgraph = self._create_sub_graph()
        self.state = None
        self.node_dossiers = {}
        self.chain_txt = ""
        # join the found  nodes into a one list
        self.all_found_nodes =[]
        self.mermaid_code = ""

    def _data_extractor_for_planner_node(self, state: ExecutionState) -> dict:
        print(f"\n[DATA EXTRACTOR] Iteration {state['iteration_count']}")
        schema = self.neo4j_connector.get_complete_schema_definition()
        all_nodes = self.tool_belt.list_all_nodes()
        print(f"[DATA EXTRACTOR] Extracting Nodes from User Goal...")
        keyword_picker = ChatPromptTemplate.from_template("""
        You are an Altair Motion Solve MBD Entity Extractor. 
        Your job is to extract search terms from a user query that will be used to find nodes in a MotionSolve Knowledge Graph.
        
        The Graph contains the following node types:
        - Bodies (Physical parts)
        - Joints (Physical parts that connect two other Physical parts)
        - PostRequests & OutputComponents (Measurements of Physical quantities)
        - StateEquation (Tire models)
        - Forces (Nodes representing a interface system to the calling Motion Solve solver in MBD model. StateEquation can use it to apply forces on a body and motion solve can read these forces as outputs of StateEquation)
        User Query will be related to one or more of these nodes. Here is the list of all nodes in the graph:
        {all_nodes}
        
        Your Rules:
        1. Extract ALL the keywords from the user query that can be used to find nodes in the graph.
        2. Give single words only. Split keywords if they are more than one word into multiple single words. Because user may be referring to multiple nodes.
        3. Ignore verbs like "check", "analyze", or "calculate".
        4. Output the result as a simple JSON list of strings.

        User Query: {goal}
        
        **Your Output (respond ONLY with a valid JSON object):**
        {{
            "keywords": [ "keyword1", "keyword2" ]
        }}
        """)           
        chain = keyword_picker | self.llm | JsonOutputParser()
        response = chain.invoke({
            "goal": state["goal"].lower(),
            "all_nodes": all_nodes.lower()
        })      
        keywords = response.get("keywords", [])
        kw_node_pairs = {}
        for kw in keywords:
            nodes=self.neo4j_connector.find_nodes_hybrid(kw.lower(), mode="hybrid", top_k=1)
            kw_node_pairs[kw] = nodes
        
        for kw, nodes in kw_node_pairs.items():
            if not nodes:
                continue
            for node in nodes:                
                if node:
                    self.all_found_nodes.append(node['name'])
        self.all_found_nodes = list(set(self.all_found_nodes))  # Unique nodes
        print(f"[DATA EXTRACTOR] Nodes extracted: {self.all_found_nodes}")        
        
        return state
    def _get_sub_graph(self, state: ExecutionState) -> dict:
        self.chain_txt,chains = self.neo4j_connector.find_chains_between_nodes(self.all_found_nodes)
        self.mermaid_code = self.neo4j_connector.generate_mermaid_topology(chains)
        
    def _planner_node(self, state: ExecutionState) -> dict:
        print(f"[DATA EXTRACTOR] Extracting Sub graphs from Nodes...")
        self._get_sub_graph(state)
        state["schema"] = self.mermaid_code
        print(f"[PLANNER] Planning action steps from Sub graphs...")
        # reset all states to start fresh
        state["plan"] = []
        state["iteration_count"] = 0
        state["next_step"] = ""
        state["past_steps"] = []
        state["last_plot"] = ""
        state['hypothesis'] = ""
        state["next_action"] = "continue"
        """
        for ch in chains.values():
            if ch.type == 'HAS_COMPONENT':
                continue
            s = ch.start_node
            if s['name'] not in self.node_dossiers:
                self.node_dossiers[s['name']] = self.tool_belt.get_enriched_dossier(s['name'])
            e = ch.end_node
            if e['name'] not in self.node_dossiers:
                self.node_dossiers[e['name']] = self.tool_belt.get_enriched_dossier(e['name'])
        """
        prompt = ChatPromptTemplate.from_template("""
            You are a Lead MBD Architect.
            The user has tasked you with the `GOAL` which is about a MotionSolve model depicted in terms of its knowledge graph equivalent MODEL CHAIN (SCHEMA).
            GOAL: {goal}
            MODEL CHAIN (SCHEMA): {schema}            
            The Graph contains the following node types:
            - Body (Physical parts)
            - Joint (Physical parts that connect two other Physical parts)
            - PostRequest & OutputComponent (Exist as pairs only. Measure Physical quantities from any node storing them time series data.)
            - StateEquation (Tire models:  Calculates force and moment based on inputs from the connected Body. Applies force on a body by sending its outputs via a interface system  node 'Force'.)
            - Force (Nodes representing a interface system to the calling Motion Solve solver in MBD model.)
            
            INSTRUCTIONS:
            1. Look at the "MODEL CHAIN" and identify every node that is NOT of type 'OutputComponent'. For each of these, create a action step to use 'dossier_digestor' to understand its physics and constraints from the goal point of view.
            2. Look at the "MODEL CHAIN" and identify pairs of nodes of type 'PostRequest' and 'OutputComponent'. Create a action step for each of these to use 'python_analysis' to generate insights from the goal point of view. 
            3. Every step in the "plan" MUST be a standalone instruction naming the specific node. Add details on how each step contributes to understanding or solving the GOAL.
            - BAD STEP: "Analyze the bodies."
            - GOOD STEP: "Use tool 'dossier_digestor' on 'node_name' to understand blah blah..."            
            - GOOD STEP: "Use tool 'python_analysis' on PostRequest 'post_req_name' and its OutputComponent 'output_component_name' to understand blah blah..."
            HYPOTHESIS:
            Based on the directional relationships in the MODEL CHAIN (e.g. Node A -> Node B), write a one-sentence engineering hypothesis about how these nodes influence the GOAL.

            OUTPUT FORMAT:
            You must respond ONLY with a JSON object. No conversation.
            {{
                "hypothesis": "Your engineering prediction based on the chain.",
                "plan": [
                    "Detailed step 1 naming specific node...",
                    "Detailed step 2 naming specific node...",
                    "..."
                ]
            }}
        """)
        
        # Simple structured output parsing
        chain2 = prompt | self.llm | JsonOutputParser()
        response = chain2.invoke({
            "goal": state["goal"],
            "schema": self.chain_txt,
        })
        state["schema"] = self.mermaid_code
        state["plan"] = response.get("plan", [])
        state["hypothesis"] = response.get("hypothesis", "")
        return state
    def _executor_node(self, state: ExecutionState) -> dict:
        current_task = state["next_step"] or state["plan"][0]
        print(f"[EXECUTOR] Task: {current_task}")

        @tool
        def fuzzy_search(approximate_node_name: str) -> List:
            """Use to find the actual names of entities and not entity types in the graph.
            Args:
                approximate_node_name (str): Approximate name of the node to search for.
            Returns: List of matching nodes."""
            return self.tool_belt.fuzzy_search_node(approximate_node_name, self.state["plan"][0])

        @tool
        def dossier_digestor(node_name: str) -> str:
            """Explains the physics and documentation of a node. Do not send the type of the node. The types are Body, Joint, Force, StateEquation, PostRequest, OutputComponent.
            Args:
                node_name (str): Exact name of the node.
            Returns: Enriched dossier string.
            """
            raw_dossier = self.node_dossiers.get(node_name) or self.tool_belt.get_enriched_dossier(node_name)
            # Prepare context
            dossier_digestor = ChatPromptTemplate.from_template("""
            "You are a Senior MBD Systems Engineer. 
            Your task is to extract "Analytical Facts" from a raw node dossier and answer the current task at hand. 
            Keep in mind that the task is focused on understanding how this node contributes to the overall model behavior with respect to the GOAL.
            The user is investigating a specific GOAL. You must ignore any information that does not contribute to understanding or solving that GOAL.

            CONTEXT:
            CURRENT TASK: {task}
            FINAL GOAL: {goal}
            OVERALL MODEL CHAIN: {chain_txt}

            RAW DOSSIER DATA:
            {raw_dossier}

            EXTRACTION RULES:
            1. IGNORE: Internal IDs, software versioning, or anything that doesnt add value to your analysis.
            2. FOCUS: On physical properties, relationships to other entities, and any parameters that could influence system's physics. Retain numerical data from node properties as much as necessary.

            OUTPUT FORMAT:
            Respond only with a JSON object in this format:
            {{
                "entity": "Name of the entity",
                "mbd_role": "Short description of its role in this specific chain",
                "key_parameters": {{ "param_name": "value", ... }},
                "analysis_impact": "How this specific node influences the GOAL in detail"
            }}
            """)
            chain = dossier_digestor | self.llm | JsonOutputParser()
            response = chain.invoke({
                "goal": self.state["goal"],
                "chain_txt": self.chain_txt,
                "raw_dossier": raw_dossier,
                "task": self.state["next_step"]
            })
            return response
        
       
        @tool
        def get_neighbors(node_name: str) -> str:
            """Use to Explores connections between entities: (node)-[rel]->(neighbor)
            (e.g., finding which PostRequest measures a Body).
            Args:
                node_name (str): Exact name of the node.
            Returns: List of neighboring nodes and their relationships.
            """
            new_nodes = self.tool_belt.get_node_neighbors(node_name)
            all_found_nodes = self.all_found_nodes
            for node in new_nodes:
                if node and node not in all_found_nodes:
                    all_found_nodes.append(node)
            self.all_found_nodes = list(set(all_found_nodes))  # Unique nodes

        @tool
        def python_analysis(cypher: str) -> str:
            """Analyzes numerical time-series data from OutputComponent nodes.

            Args:
                cypher (str): Cypher query to fetch data. 
                Example: MATCH (pr:PostRequest {name: 'post_req_name'})-[:HAS_COMPONENT]->(oc:OutputComponent {name: 'OC_name'}) 
                        RETURN oc.time_values AS time, oc.output_values AS val
            """
            # Prepare context
            code_writer = ChatPromptTemplate.from_template("""
            "You are a Senior MBD Numerical Analyst.
            You need to write a python code to analyze the time-series data from MotionSolve OutputComponents 
            You DONT have to Answer the CURRENT TASK. 
            Understand what the TASK requires with the bigger GOAL in mind. 
            Write python code to get the insights required to answer the TASK.
            
            TASK: {task}
            GOAL: {goal}
            Neo4j Context: {chain_txt}

            INPUT DATA: You have a pandas DataFrame named `df` with two columns:
            - 'time': List of time steps.
            - 'val': List of numerical values for the signal.            
            LIBRARIES: pandas (pd), numpy (np), scipy (sp), plotly.graph_objects (go). 
            These are availble to in your local scope as per the alias given above.
            REQUIREMENTS:
            1. Perform numerical analysis relevant to the TASK using the data in `df`.
            2. Use all necessary numerical methods and statistical techniques.
            3. Use all standard numerical guards (e.g., handling NaNs, divide by zero etc...).
            OUTPUT:
            1. Print a detailed summary of your findings as they relate to the TASK. 
            2. Create an interactive Plotly figure named `plotly_fig` using `plotly.graph_objects`.
            MANDATORY:
            1. Create an interactive Plotly figure named `plotly_fig` using `plotly.graph_objects`.
            2. Do NOT call `plotly_fig.show()` or `plt.show()`. Just define the `plotly_fig` object.
            3. Use all standard numerical guards 
            4. Do NOT include any markdown formatting or '```python' tags. Just raw code.""")
            llm_gpt_4o = ChatOpenAI(model=global_vars.model_openai_4o, temperature=0)
            chain = code_writer | llm_gpt_4o | StrOutputParser()
            generated_code = chain.invoke({
                "goal": self.state["goal"],
                "chain_txt": self.chain_txt,                
                "task": self.state["next_step"]
            })
            cleaned_code = generated_code.replace("```python", "").replace("```", "").strip()
            py_data = self.tool_belt.run_python_analysis(cypher, cleaned_code)
            # get only text summary and the plotly json from the result
            Plotly_JSON = py_data.get("Plotly_JSON","")
            py_data = py_data.get("Analysis_Result","")
            data_analyst = ChatPromptTemplate.from_template("""
                "You are a Senior MBD Analyst. You need to answer the CURRENT TASK using the PYTHON ANALYSIS RESULTS provided. Ignore things that are not relevant to the TASK.
                The CURRENT TASK is part of a bigger GOAL. This is given to give a context of why this TASK was created.
                TASK: {task}
                BIGGER GOAL: {goal}
                Neo4j Context: {chain_txt}
                PYTHON ANALYSIS RESULTS:{py_data}
                Respond only with a JSON object in this format:
                {{
                    "entity": "Name of the OutputComponent and PostRequest analyzed. Pick it from task",
                    "mbd_role": "Short description of their role in this specific chain",
                    "key_parameters": {{ "param_name": "value", ... }},
                    "analysis_impact": "How this specific output data influence the GOAL in detail"
                }}
            """)
            data_analyst_chain = data_analyst | self.llm | JsonOutputParser()
            analysis_summary = data_analyst_chain.invoke({
                "goal": self.state["goal"],
                "chain_txt": self.chain_txt,                
                "task": self.state["next_step"],
                "py_data": py_data
            })
            analysis_summary['Plotly_JSON'] = Plotly_JSON
            return analysis_summary

        tools = [dossier_digestor, get_neighbors, python_analysis]
        
        llm_with_tools = self.llm.bind_tools(tools)
        self.state=state
        tool_selector_prompt = ChatPromptTemplate.from_template("""
        "You are an MBD router agent. Your role is to only Choose the tool mentioned in the task. Dont interpret the task. Just pick the tool mentioned.
        Task: {current_task}. """)
                                                                
        tool_selector_exec = tool_selector_prompt | llm_with_tools 
        response = tool_selector_exec.invoke({
            "current_task": current_task
        })
        formatted_findings = []
        new_entities = {}
        if response.tool_calls:
            for call in response.tool_calls:
                t_name = call["name"]
                args = call["args"]
                # Execution Mapping
                tool_map = {
                    "dossier_digestor": dossier_digestor,
                    "python_analysis": python_analysis,
                    "get_neighbors": get_neighbors
                }
                # Execute Tool
                if t_name in tool_map:
                    raw_result  = tool_map[t_name].invoke(args)
                    if isinstance(raw_result, dict):
                    # For the Dossier Dictionary:
                    # We convert to a pretty-printed string for the history
                    # For the Python Analysis Dictionary:
                        if raw_result.get("Plotly_JSON"):                        
                            state["last_plot"] = raw_result.get("Plotly_JSON")
                            del raw_result["Plotly_JSON"]
                        content = json.dumps(raw_result, indent=2)
                        header = f"ENTITY METADATA (Source: {t_name})"
                    else:
                        # For the Python Analysis String:
                        content = str(raw_result)
                        header = f"(Source: {t_name})"
                    if t_name == 'get_neighbors':
                        state['next_action'] = 'replan'
                    else:
                        state['next_action'] = 'continue'

                    # Wrap in a clear block so the LLM sees the boundary
                    formatted_findings.append(f"### {header}\n{content}")

            # Combine all results from this task into one "Step Entry"
            combined_entry = (
                f"TASK: {current_task}\n"                
                f"RESULTS:\n" + "\n\n".join(formatted_findings)
            )
        else:
            combined_entry=(
                f"TASK: {current_task}\n"                
                f"RESULTS:\n" + "\n\n".join(response.content)
            )
        
        state ["past_steps"]= [combined_entry]        
        state["iteration_count"] += 1
        state["next_step"] = state["plan"][state["iteration_count"]] if state["iteration_count"] < len(state["plan"]) else ""
        return state
    
      
    def _summarizer_node(self, state: ExecutionState) -> dict:
        prompt = ChatPromptTemplate.from_template("Analyse and Summarize `Findings` for goal: {goal}" \
        "Findings: {history}" \
        "Relevant Neo4j Context from MBD Model: {chain_txt}" \
        "Your Original Hypothesis which you made before getting the Findings: {hypothesis}" \
        "Provide a detailed final answer. Use data and facts to support your conclusions.")
        chain = prompt | self.llm
        res = chain.invoke({"goal": state["goal"], 
                            "history": state["past_steps"],
                            "chain_txt": self.chain_txt,
                            "hypothesis": state.get("hypothesis", "")})
        return {"final_answer": res.content}

    def _create_sub_graph(self):
        workflow = StateGraph(ExecutionState)
        workflow.add_node("data_extractor", self._data_extractor_for_planner_node)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("summarizer", self._summarizer_node)
        
        workflow.set_entry_point("data_extractor")
        workflow.add_edge("data_extractor", "planner")
        workflow.add_edge("planner", "executor")
        
        workflow.add_conditional_edges(
            "executor",
            # route to planner if 'next_action' is 'replan' else if 'continue' and there is a next step go to executor else to summarizer
            lambda s: "replan" if s["next_action"] == "replan" else "execute" if  s["next_step"] else "summarize",
            {"execute": "executor", "replan": "planner", "summarize": "summarizer"}
        )
        workflow.add_edge("summarizer", END)
        return workflow.compile(checkpointer=MemorySaver())

    def run(self, query: str):
        config = {"configurable": {"thread_id": "1"}}
        inputs = {"goal": query, "plan": [], "past_steps": [], "discovered_entities": {}, "iteration_count": 0}
        for event in self.subgraph.stream(inputs, config):
            print(event)
    # --- ENTRYPOINT ---
    def parse_latest_finding(self, finding_str: str, task_no: int) -> str:
        """Extracts a clean summary for the UI status bar."""
        if "RESULTS:" not in finding_str:
            return "Processing..."

        # 1. Isolate the results section
        results_part = finding_str.split("RESULTS:")[1].strip()

        # 2. Case A: It's a JSON Metadata block (Dossier Digest)
        if "### ENTITY METADATA" in results_part:
            try:
                # Find the first { and last } to isolate the JSON string
                json_match = re.search(r"(\{.*\})", results_part, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(1))
                    # Return the specific insight
                    return f"🔍 Insight of Task {task_no}: {data.get('analysis_impact', 'Node analysis complete.')}"
            except Exception:
                pass
        return "Thinking..."
    def process_message(self, message: str, chat_history: List[BaseMessage]) -> Generator[dict, None, None]:
        """Runs the graph and yields structured data for the UI."""
        config = {"configurable": {"thread_id": "1"}}
        # Ensure your state has 'schema' to store the chain_txt
        inputs = {
            "goal": message, 
            "plan": [], 
            "next_step": "", 
            "past_steps": [], 
            "schema": "", 
            "iteration_count": 0,
            "next_action": "continue",
        }
        
        for output in self.subgraph.stream(inputs, config=config):
            for key, value in output.items():
                
                # 1. Update the Chain/Topology Tab
                if "schema" in value and value["schema"]:
                    yield {"type": "chain", "data": value["schema"]}

                # 2. Update the Plan Checklist
                if "plan" in value and value["plan"]:
                    yield {"type": "plan", "data": value["plan"]}

                # 3. Status updates from the Executor
                if key == "executor":
                    # Get the most recent finding to show as status
                    latest_finding = value['past_steps'][-1] if value['past_steps'] else "Thinking..."
                    clean_status = self.parse_latest_finding(latest_finding, value["iteration_count"])
                    yield {"type": "status", "data": clean_status}
                    
                    # Check if a plot was generated
                    if "last_plot" in value and value["last_plot"]:
                        yield {"type": "plot", "data": value["last_plot"]}

                # 4. The Final Conclusion
                elif key == "summarizer":
                    if "final_answer" in value:
                        yield {"type": "text", "data": value["final_answer"]}
    
        
    def save_graph(self, filepath: Path):
            import requests
            """Saves the graph structure to a file."""
            graph = self.subgraph.get_graph()
            try:
                # Draw the graph and save it as a PNG file
                # You can also use .draw_svg() or .draw_mermaid() for other formats
                image_data = graph.draw_mermaid()
                content = base64.b64encode(image_data.encode('utf-8')).decode('utf-8')
                url = f"https://mermaid.ink/img/{content}"
                
                # 3. Download the image from the working URL
                response = requests.get(url)
                #print(f"\nClick here to see your graph: {url}")
                # Save the image data to a file
                with open(filepath, "wb") as f:
                    f.write(response.content)

                print(f"✅ Graph visualization saved to {filepath}")
            except Exception as e:
                print(f"❌ Could not visualize graph. Make sure you have installed graphviz.")
                print(f"   Error: {e}")

if __name__ == "__main__":
    user_query = 'Analyze the properties of the hub body to understand its characteristics, mass distribution, and any constraints. Use a Cypher query to extract the hub body node and its relevant properties.'   
    qa_action_exec_agent = ActionStepExecutorAnalyst()
    qa_action_exec_agent.save_graph(Path(__file__).parent / "qa_action_exec_agent_h_graph.png")        