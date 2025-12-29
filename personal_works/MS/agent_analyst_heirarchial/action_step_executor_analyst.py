import operator
from typing import List, TypedDict, Annotated, Dict, Optional,Generator
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser

from agent_tools_2 import ToolBelt
from pathlib import Path

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
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.neo4j_connector = connector
        self.tool_belt = ToolBelt(connector,self.llm)
        self.subgraph = self._create_sub_graph()
        self.state = None

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
        - AutoTireSystems (Tire models)
        - Forces (Nodes representing a interface system to the calling Motion Solve solver in MBD model. AutoTireSystems can use it to apply forces on a body and motion solve can read these forces as outputs of AutoTireSystems)
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
        chain_txt,chains = self.neo4j_connector.find_chains_between_nodes(all_found_nodes)
        for ch in chains.values():
            if ch.type == 'HAS_COMPONENT':
                continue
            s = ch.start_node
            s_dossier = self.tool_belt.get_enriched_dossier(s['name'])
            e = ch.end_node
            e_dossier = self.tool_belt.get_enriched_dossier(e['name'])
        prompt = ChatPromptTemplate.from_template("""
        You are a Lead MBD Analyst. You need to answer the goal with a deep analysis.         
        GOAL: {goal}
        SCHEMA: {schema}        
        PAST FINDINGS: {past_steps}        
                                                  
        STRATEGY:
        Based on a sentimental and keyword analysis of the goal with respect to the knowledge graph of the correpsonding MotionSolve MBD model in question, a causal chain of graph nodes "SCHEMA" has been identified that are relevant to the goal.
        Information on each of these nodes and the relationships between them is waiting to be gathered.
        Using these nodes, two things are required from you finally in the form of json output:
        1. create hypothesis: answer the goal based on a analysis of the schema and your knowledge of MBD principles.
        2. explain hypothesis: devise a step-by-step plan to achieve the hypothesis. Each step should be clear and actionable.
        Steps to follow to achieve your output
        1. understand the various entitites involved in the goal. you can use the tool 'get_dossier' to get detailed information of each node and its properties as present in the MBD model.
        2. understand their relationships. The chain already gives you a directional relationship between the nodes. (node)-[rel]->(neighbor). You can use get_dossier tool to get more information on the relationships as well.
        3. Get the smaller details and the bigger picture and answer the goal.        
        4. use 'python_analysis' tool to run analysis on the data from outputcomponent type of nodes. OutputComponent has to be used along with PostRequest nodes only as they form a pair.
        We will finally revisit to give a detailed answer to the goal based on all the findings.                                                                                           
        Respond with a JSON list of remaining steps. If finished, return an empty list.
        **Your Output (respond ONLY with a valid JSON object):**
        {{
            "plan": [ "step1", "step2", "... ],
            "hypothesis": "your hypothesis here",
        }}
              
        """)
        
        # Simple structured output parsing
        chain2 = prompt | self.llm | JsonOutputParser()
        response = chain2.invoke({
            "goal": state["goal"],
            "schema": chain_txt,
            "past_steps": "\n".join(state["past_steps"][-3:]), # Only last 3 for focus,            
        })
        
        # In production, use .with_structured_output. Here we simulate for brevity.
        # This part assumes the LLM returns a clean plan.        

        return {"plan": response.get("plan")}

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
        def get_dossier(node_name: str) -> str:
            """Explains the physics and documentation of a node.
            Args:
                node_name (str): Exact name of the node.
            Returns: Enriched dossier string.
            """
            return self.tool_belt.get_enriched_dossier(node_name)
        
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
        def python_analysis(cypher: str, code: str) -> str:
            """Use to process numerical data once the Cypher query is known."""
            return self.tool_belt.run_python_analysis(cypher, code)

        @tool
        def list_nodes_by_type(node_type: str) -> str:
            """Lists all nodes of a given type such bodies, joints, or requests in the graph."
            Args:
                node_type (str): Type of nodes to list.
            Returns: List of node names.
            """
            return self.tool_belt.list_nodes_by_type(node_type)
        tools = [fuzzy_search, get_neighbors, python_analysis, fetch_data, list_nodes_by_type]
        llm_with_tools = self.llm.bind_tools(tools)
        self.state=state
        schema = self.neo4j_connector.get_complete_schema_definition()
        response = llm_with_tools.invoke(f"Task: {current_task}. Schema:{schema}. Context: {state['discovered_entities']}")
        
        new_findings = []
        new_entities = {}
        executore_result = ""
        if response.tool_calls:
            for call in response.tool_calls:
                t_name = call["name"]
                args = call["args"]
                
                # Execute Tool
                if t_name == "fuzzy_search":
                    res = fuzzy_search.invoke(args)
                    if "matches" in res:
                        # Extract the first match as the "source of truth"
                        match = res["matches"]
                        new_entities[args["approximate_node_name"]] = match["name"]
                    new_findings.append(str(res))
                elif t_name == "fetch_data":
                    res = fetch_data.invoke(args)
                    new_findings.append(res)
                elif t_name == "get_neighbors":
                    res = get_neighbors.invoke(args)
                    new_findings.append(res)
                elif t_name == "list_nodes_by_type":
                    res = list_nodes_by_type.invoke(args)
                    new_findings.append(res)
                elif t_name == "python_analysis":
                    res = python_analysis.invoke(args)
                    new_findings.append(res)

                elif t_name == "get_dossier":
                    res = get_dossier.invoke(args)
                    new_findings.append(res)
            executore_result = self._result_analyser_node(current_task, new_findings)
        else:
            new_findings.append(response.content)
        
        state ["past_steps"]= [f"Task: {current_task}\nResult: {executore_result}"]
        state["discovered_entities"] = new_entities
        state["iteration_count"] = state["iteration_count"] + 1
        self._plan_updater_node(state)
        return state
    def _plan_updater_node(self, state: ExecutionState) -> dict:
        """Updates the plan by removing the executed step."""
        plan_updater = ChatPromptTemplate.from_template("""
        You are an expert MBD Analyst. Based on final goal, a plan was devised, few of the steps have been executed steps and you have its results, tell what to do next.
        Goal: {goal}
        Past Steps and Results: {past_steps}        
        **Your Output (respond Only with a concise next step) Dont include any explanations or additional text based on your analysis. Just give the next step:**
        """)
        chain = plan_updater | self.llm | StrOutputParser()
        next_step = chain.invoke({
            "goal": state["goal"],
            "past_steps": "\n".join(state["past_steps"])
        })
        next_step = next_step.strip()
        state["next_step"] = next_step
        return state
    
    def _result_analyser_node(self, current_task, new_findings) -> dict:
        task_finding_analyser = ChatPromptTemplate.from_template("""
        You are an expert MBD Analyst. Based on the task you just performed and its results, summarize the key findings concisely.
        Task: {current_task}
        Results: {new_findings}
        **Your Output (respond with concise findings):**
        """)
        chain = task_finding_analyser | self.llm | StrOutputParser()
        findings_summary = chain.invoke({
            "current_task": current_task,
            "new_findings": "\n".join(new_findings)
        })
        findings_summary = findings_summary.strip()
        return findings_summary
      
    def _summarizer_node(self, state: ExecutionState) -> dict:
        prompt = ChatPromptTemplate.from_template("Summarize results for goal: {goal}\nFindings: {history}")
        chain = prompt | self.llm
        res = chain.invoke({"goal": state["goal"], "history": state["past_steps"]})
        return {"final_answer": res.content}

    def _create_sub_graph(self):
        workflow = StateGraph(ExecutionState)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("summarizer", self._summarizer_node)
        
        workflow.set_entry_point("planner")
        
        workflow.add_conditional_edges(
            "planner",
            lambda s: "summarize" if not s["plan"] or s["iteration_count"] > 10 else "execute",
            {"execute": "executor", "summarize": "summarizer"}
        )
        workflow.add_edge("executor", "planner")
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