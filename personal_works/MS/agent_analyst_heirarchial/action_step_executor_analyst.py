import operator
from typing import List, TypedDict, Annotated, Dict, Optional,Generator
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
import json

from agent_tools_2 import ToolBelt
from pathlib import Path
import global_vars
# --- STATE DEFINITION ---
class ExecutionState(TypedDict):
    goal: str
    plan: List[str]
    past_steps: Annotated[List[str], operator.add]
    discovered_entities: Dict[str, str] # CRITICAL: Stores {"hub": "Hub_Body_01"}
    iteration_count: int
    final_answer: str
    next_step: Optional[str]

    
class ActionStepExecutorAnalyst:
    def __init__(self, connector):
        self.llm = ChatOpenAI(model=global_vars.model_openai_4omini, temperature=0)
        self.neo4j_connector = connector
        self.tool_belt = ToolBelt(connector,self.llm)
        self.subgraph = self._create_sub_graph()
        self.state = None
        self.node_dossiers = {}
        self.chain_txt = ""
    def _planner_node(self, state: ExecutionState) -> dict:
        print(f"\n[PLANNER] Iteration {state['iteration_count']}")
        dead_loop = False
        if len(state["past_steps"]) > 2:
            last_two = state["past_steps"][-2:]
            if last_two[0] == last_two[1]:
                print("!!! LOOP DETECTED - FORCING STRATEGY SHIFT !!!")
                dead_loop = True
        schema = self.neo4j_connector.get_complete_schema_definition()
        all_nodes = self.tool_belt.list_all_nodes()
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
        # join the found  nodes into a one list
        all_found_nodes =[]
        for kw, nodes in kw_node_pairs.items():
            if not nodes:
                continue
            for node in nodes:                
                if node:
                    all_found_nodes.append(node['name'])
        all_found_nodes = list(set(all_found_nodes))  # Unique nodes
        print(f"[PLANNER] Extracted Nodes: {all_found_nodes}")
        self.chain_txt,chains = self.neo4j_connector.find_chains_between_nodes(all_found_nodes)
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
            You are a Lead MBD Architect. Your goal is to solve a specific engineering problem using a MotionSolve model.
            
            GOAL: {goal}
            MODEL CHAIN (SCHEMA): {schema}            

            INSTRUCTIONS:
            1. Look at the "MODEL CHAIN" and identify every node that is NOT an 'OutputComponent'. For each of these, create a plan step to use 'dossier_digestor' to understand its physics and constraints from the goal point of view.
            2. Look at the "MODEL CHAIN" and identify 'PostRequest' and 'OutputComponent' pairs. Create a specific plan step to use 'python_analysis' for these pairs. State clearly the node type also so as to create cypher query easily.
            3. Every step in the "plan" MUST be a standalone instruction naming the specific node. Add details on how each step contributes to understanding or solving the GOAL.
            - BAD STEP: "Analyze the bodies."
                        
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
        
        # In production, use .with_structured_output. Here we simulate for brevity.
        # This part assumes the LLM returns a clean plan.        

        return {"plan": response.get("plan"), "next_step":response.get("plan")[0],"iteration_count": 0}

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
            """Explains the physics and documentation of a node.
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
        def fetch_data(node_name: str) -> str:
            """Confirms data exists and gets basic stats before full analysis.
            Args:
                node_name (str): Exact name of the PostRequest or OutputComponent node.
            Returns: Data statistics string.
            """            
            return self.tool_belt.fetch_numerical_data(node_name, self.state["plan"][0])
        
        @tool
        def get_neighbors(node_name: str) -> str:
            """Use to Explores connections between entities: (node)-[rel]->(neighbor)
            (e.g., finding which PostRequest measures a Body).
            Args:
                node_name (str): Exact name of the node.
            Returns: List of neighboring nodes and their relationships.
            """
            return self.tool_belt.get_node_neighbors(node_name)

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

            LIBRARIES: pandas (pd), numpy (np), scipy (sp).
            REQUIREMENTS:
            1. Perform numerical analysis relevant to the TASK using the data in `df`.
            2. Print a detailed summary of your findings as they relate to the TASK. This is going to run in a isolated sandbox. So dont plot anything. PRINT your required findings only.
            3. Do NOT include any markdown formatting or '```python' tags. Just raw code.""")
            llm_gpt_4o = ChatOpenAI(model=global_vars.model_openai_4o, temperature=0)
            chain = code_writer | llm_gpt_4o | StrOutputParser()
            generated_code = chain.invoke({
                "goal": self.state["goal"],
                "chain_txt": self.chain_txt,                
                "task": self.state["next_step"]
            })
            cleaned_code = generated_code.replace("```python", "").replace("```", "").strip()
            py_data = self.tool_belt.run_python_analysis(cypher, cleaned_code)
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
            return analysis_summary

        @tool
        def list_nodes_by_type(node_type: str) -> str:
            """Lists all nodes of a given type such bodies, joints, or requests in the graph."
            Args:
                node_type (str): Type of nodes to list.
            Returns: List of node names.
            """
            return self.tool_belt.list_nodes_by_type(node_type)
        tools = [dossier_digestor, get_neighbors, python_analysis]
        
        llm_with_tools = self.llm.bind_tools(tools)
        self.state=state
        tool_selector_prompt = ChatPromptTemplate.from_template("""
        "You are an MBD execution agent. Choose the correct tool to complete the task.
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
                    "fuzzy_search": fuzzy_search,
                    "dossier_digestor": dossier_digestor,
                    "python_analysis": python_analysis,
                    "fetch_data": fetch_data,
                    "get_neighbors": get_neighbors
                }
                # Execute Tool
                if t_name in tool_map:
                    raw_result  = tool_map[t_name].invoke(args)
                    if isinstance(raw_result, dict):
                    # For the Dossier Dictionary:
                    # We convert to a pretty-printed string for the history
                        content = json.dumps(raw_result, indent=2)
                        header = f"ENTITY METADATA (Source: {t_name})"
                    else:
                        # For the Python Analysis String:
                        content = str(raw_result)
                        header = f"NUMERICAL ANALYSIS (Source: {t_name})"

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
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("summarizer", self._summarizer_node)
        
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")
        
        workflow.add_conditional_edges(
            "executor",
            lambda s: "summarize" if not s["next_step"] or s["iteration_count"] > 10 else "execute",
            {"execute": "executor", "summarize": "summarizer"}
        )
        workflow.add_edge("summarizer", END)
        return workflow.compile(checkpointer=MemorySaver())

    def run(self, query: str):
        config = {"configurable": {"thread_id": "1"}}
        inputs = {"goal": query, "plan": [], "past_steps": [], "discovered_entities": {}, "iteration_count": 0}
        for event in self.subgraph.stream(inputs, config):
            print(event)
    # --- ENTRYPOINT ---

    def process_message(self, message: str, chat_history: List[BaseMessage]) -> Generator[str, None, None]:
        """Standard method to run the graph."""
        config = {"configurable": {"thread_id": "1"}}
        inputs = {"goal": message, "plan": [], "next_step": "", "past_steps": [], "discovered_entities": {}, "iteration_count": 0}
        
        final_answer_started= False
        for output in self.subgraph.stream(inputs, config=config):
            # We can yield status updates to the UI here
            for key, value in output.items():
                if key == "executor":
                    yield f"STATUS: {value['past_steps']}\n"
                elif key == "summarizer":
                    final_answer_started = True
                    if value['final_answer'] and not final_answer_started:
                        yield "FINAL_ANSWER_START\n"
                    if value['final_answer']:
                        yield f"\nFINAL ANSWER:\n {value['final_answer']}"
    
        
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