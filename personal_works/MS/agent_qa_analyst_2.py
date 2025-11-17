# agent_qa_test_analyst.py
from pathlib import Path
from typing import List, TypedDict, Generator, Optional, Any
from operator import itemgetter
import re
from urllib import response

import global_vars
from neo4j_kg_builder import Neo4jConnector

from langchain_core.messages import BaseMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.documents import Document
from pydantic.v1 import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader

from langgraph.graph import StateGraph, END

from agent_tools_2 import ToolBelt

class GraphState(TypedDict):
    node_types: list[str]
    node_names: list[str]
    hypotheses: List[str]  # A ranked list of potential root causes
    reasoning_log: List[str]
    question: str
    chat_history: Optional[List[BaseMessage]]
    answer: str
    plan: str
    plan_step: int
    step_summaries: List[str]
    step_decisions: List[str]


class KG_Query(BaseModel):
    query_by_type: List[str] = Field(..., description="A list of node types to retrieve from the KG.")
    query_by_name: List[str] = Field(..., description="A list of node names to retrieve from the KG.")

class ActingPlanRewriteOutput(BaseModel):
    rewritten_hypotheses: List[str] = Field(..., description="The rewritten hypotheses as a ranked list.")
    rewritten_plan: List[str] = Field(..., description="The rewritten action plan as a list of steps.")
    summary: str = Field(..., description="A summary justifying the decision. Decision can be the rewrites made to the action plan or the decision explaining to end the investigation early.")

class QAAnalystAgent:
    """An agent that investigates issues by querying a Neo4j knowledge graph."""
    def __init__(self):
        self.EXPERT_MINDSET_TEXT = """
        Bodies are connected to other bodies through Joints. 
        Motions are applied to Joints to drive the system, 
        or can come from direct Initial Conditions on Bodies. 
        Initial conditions can be given directly on the Body properties or 
        through user-controlled subroutines like SETIC/SETWIC. 
        Motions result in changes to the position, velocity, and acceleration of the Bodies in translation and rotation.
        State variables track these changes over time by measuring them using markers attached to bodies. 
        This is used by state equations if present to obtain the kinematics of the current state, which is used as inputs to state equations.
        State equations perform calculations to output forces and moments that act on bodies, joints, or other entities in the system as per their connections.                
        These Forces and moments can be calulated by simple expressions or as constants or calculated via user subroutines employing state equations. 
        These forces, in turn, affect the Bodies' motion, creating a feedback loop that evolves over time. 
        PostRequests provide a way to measure and plot these physical quantities for analysis by user. Post requests measure displacements, velocities, accelerations, forces, moments, and other derived quantities at specified locations using markers attached to bodies in the model during the simulation.
        Whenever a general symptom is reported, postrequests are often the best starting point to trace back to root causes. Identify post requests relevant to the symptom, then trace their dependencies back through forces, motions, joints, and bodies to find potential issues.
        """
        # Initialize Neo4j connector
        self.neo4j_connector = Neo4jConnector(
            uri=global_vars.NEO4J_URI,
            user=global_vars.NEO4J_USER,
            password=global_vars.NEO4J_PASSWORD
        )
        # Initialize toolbelt
        self.tool_belt = ToolBelt(self.neo4j_connector)
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=global_vars.model_openai_4omini,
            #openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            #openai_api_key=global_vars.google_api_key,
            temperature=0.3,
            streaming=True,
        )
        self.qa_graph = self._create_graph()

    def close(self):
        self.neo4j_connector.close()

    def _rewrite_hypothesis_plan(self, state: GraphState) -> str:
        """Rewrite hypotheses into clear, distinct statements."""
        
        rewrite_prompt = ChatPromptTemplate.from_template(
            """You are a Master CAE Expert for Altair MotionSolve. Your role is to act as a supervisor, creating a strategic plan to investigate a user's query.
            **Expert Mindset (First Principles of the System):**        
            {expert_mindset} 
            
            You had already proposed hypotheses and action plans for investigating the user query.
            user query:
            {user_question}
            
            Original Hypothesis:
            {hypothesis}
            
            Original Action Plan:
            {action_plan}
            
            The investigation has started executing the action plan step by step. Some information has been gathered so far.
            information gathered so far:
            {investigation_so_far}
            
            Goal:
            Goal is to check the current status of action plan based on the new information. You can do four things: 
            1. As you have more information now, revisit the hypothesis and reframe the action plan if necessary. Feel free. You may delete or add new steps as needed. 
            2. If the new information contradicts your original hypothesis, you must rewrite it to better fit the evidence.
            3. If the original plan is still valid, you can keep it as is.
            4. If a conclusion can be drawn based on the new information, you may choose to end the investigation early. In that case, provide a empty list of action plan.
            5. Always justify your decision with a summary that includes key facts and reasoning as a standalone explanation. This summary will be used in the final answer.
            6. Remember, you are the one who will ultimately stop the investigation when you feel enough information has been gathered. You can choose to end the investigation at any step if you feel at least 70-80'%' confident in your findings.
            
            **Your Output (respond ONLY with a valid JSON object matching this schema):**
            {{
                "rewritten_hypotheses": [
                    "Rank 1: [Your primary hypothesis here]",
                    "Rank 2: [Your secondary hypothesis here]"
                ],
                "rewritten_plan": [
                "1. Your clear, actionable step 1 here.",
                "2. Your clear, actionable step 2 here.",
                "3. ...",
                list all the steps needed to gather context for the rewritten action plan. Be exhaustive and speficic. Empty if ending investigation.
                ],
                "summary": "A summary justifying the decision. Decision can be the rewrites made to the action plan or the decision explaining to end the investigation. Make it standalone with key facts that help you decide, so that it can used in final answer."
            }}
            """
        )
        rewrite_chain = rewrite_prompt | self.llm.with_structured_output(schema=ActingPlanRewriteOutput)
        rewritten = rewrite_chain.invoke({
            "expert_mindset":self.EXPERT_MINDSET_TEXT,
            "user_question": state["question"],
            "hypothesis": state["hypotheses"],
            "action_plan": '\n'.join(state["plan"]),
            "investigation_so_far": '\n'.join(state["step_summaries"])
        })
        state["plan"]=rewritten.rewritten_plan
        state["hypotheses"]=rewritten.rewritten_hypotheses
        state["step_decisions"]=state["step_decisions"] + [rewritten.summary]        
        
        print(f"---INFO: Step Decisions --- \n{state['step_decisions']}")
        return state
        
    def _create_initial_hypotheses(self, state: GraphState) -> GraphState:
        """Initial state: Determine starting nodes based on user question."""
        question = state["question"]
        motion_solve_graph_schema = self.neo4j_connector.get_complete_schema_definition()
        
        supervisor_prompt_template = """You are a Master CAE Diagnostician for Altair MotionSolve. Your role is to act as a supervisor, creating a strategic plan to investigate a user's query.
        **Expert Mindset (First Principles of the System):**        
        {expert_mindset}

        **Your Goal:**
        Your goal is to use this mindset to understand the user's question and create a plan to investigate it. The user is reporting a symptom, and you must plan to find its cause.

        **Your Task (Think Step-by-Step):**
        1. Read the "Expert Mindset" to understand how entities are causally linked.
        
        2. Read the user's question carefully. It describes a symptom in a vehicle dynamics simulation.

        **Consult the System Schema:**
        Review the available Node and Relationship types in the knowledge graph. This tells you what kinds of components and interactions exist in the model.
        {schema}

        **3. Formulate Hypotheses:**
        Based on the symptom and your expert knowledge, generate a ranked list of 2-3 distinct, testable hypotheses for the root cause. A good hypothesis links a potential cause to the observed effect.
        Example 1: "Hypothesis: Asymmetric tire forces are causing the vehicle to pull left."

        **4. Create an Initial Action Plan:**
        Define clear, concrete step by step actions to start testing the hypotheses one by one. Be very explorative and cover multiple angles. This is the beginning of a investigation, so try to attack the query from multiple perspectives.
        The action plan will be used to gather context of entities by type or name from the knowledge graph by another agent.
        **User's Question:**
        "{question}"

        **Your Output (respond ONLY with a valid JSON object):**
        {{
            "hypotheses": [
                "Rank 1: [Your primary hypothesis here]",
                "Rank 2: [Your secondary hypothesis here]"
            ],
            "plan": "[
            1. Your clear, actionable step 1 here.",
            2. Your clear, actionable step 2 here.
            3. ...
            list all the steps needed to gather context for the action plan. Be exhaustive and speficic.
            ]"
        }}
        """
        supervisor_prompt = ChatPromptTemplate.from_template(supervisor_prompt_template)
        supervisor_chain = supervisor_prompt | self.llm | JsonOutputParser()

        response = supervisor_chain.invoke({
            "question": question,
            "schema": motion_solve_graph_schema,
            "expert_mindset": self.EXPERT_MINDSET_TEXT
        })

        hypotheses = response.get("hypotheses", [])
        plan = response.get("plan", "No plan generated.")

        print(f"---INFO: Supervisor Hypotheses --- \n{hypotheses}")
        print(f"---INFO: Supervisor Plan --- \n{plan}")

        # Initialize the reasoning log with the supervisor's output
        log_entry = "Supervisor Analysis Complete.\n"
        log_entry += "Hypotheses:\n" + "\n".join([f"- {h}" for h in hypotheses])
        log_entry += f"\nInitial Plan: {plan}"
        state["node_types"] = []
        state["node_names"] = []
        return {
            "hypotheses": hypotheses,
            "plan": plan,
            "reasoning_log": [log_entry], # Start the log
            "plan_step": 0
        }
    
    def _get_step_kg_context(self, state: GraphState) -> GraphState:
        """Retrieve relevant context from the knowledge graph based on the query."""
        plan_step = state["plan_step"]
        if plan_step >= len(state["plan"]):
            return state  # No more steps to process
        current_step_description = state["plan"][plan_step]
        print(f"\n---NODE: EXECUTING PLAN STEP {plan_step + 1}: '{current_step_description}'---")
        
        motion_solve_graph_schema = self.neo4j_connector.get_complete_schema_definition()
        get_all_nodes_with_primary_type = self.neo4j_connector.get_all_nodes_with_primary_type()
        nodes=''
        for i in get_all_nodes_with_primary_type:
            if i['type'] not in ['OutputComponent','Reference_Marker']:
                nodes += f'name: {i["name"]}, type: {i["type"]}\n'
        kg_prompt_template = """You are an assistant that translates a natural language action plan step into a specific query for a knowledge graph.
            
            **Knowledge Graph Schema:**
            {schema}
            
            **Available Nodes in the Graph (Name and Type):**
            {nodes}

            **Current Action Step:**
            "{step}"
            
            Based *only* on the current action step and the available nodes, determine which nodes or types of nodes need to be queried to gather the necessary information. Keep the query focused and minimal, only what's needed for this specific step.
            
            You must respond only in the following JSON format:
            {{
                "query_by_type": ["type1", "type2"],
                "query_by_name": ["name1", "name2"]
            }}
            The entities should be relevant to the action plan and help in investigating the action plan.
            If no specific query is needed (e.g., the step is a logical deduction), return empty lists.
        """
        supervisor_prompt = ChatPromptTemplate.from_template(kg_prompt_template)
        supervisor_chain = supervisor_prompt | self.llm.with_structured_output(schema=KG_Query)

        response = supervisor_chain.invoke({
            "step": current_step_description,
            "schema": motion_solve_graph_schema,
            "nodes": nodes
        })
        log_entry = f"--- INFO: KG Query completed for step '{current_step_description}':\n"
        log_entry += str(response)
        node_types = response.query_by_type if hasattr(response, 'query_by_type') else []
        node_names = response.query_by_name if hasattr(response, 'query_by_name') else []
        state["node_types"]=node_types
        state["node_names"]=node_names
        print(log_entry)
        
        return {
            "reasoning_log": state["reasoning_log"] + [log_entry], # Start the log
            "node_types": state["node_types"],
            "node_names": state["node_names"],
        }
    
    def _get_documentation_explanation(self, node_name: str) -> str:
        """
        The "Auto-Researcher" tool. Takes a node's raw data, determines its type,
        fetches the relevant documentation, and uses an LLM to synthesize an
        explanation of that specific node's properties.
        """
        if not node_name:
            return ""
        node_data = self.neo4j_connector.get_node_properties(node_name)
        if not node_data:
            return ""
        # Determine the primary type of the node (e.g., "Joint", "Motion")
        node_type = next((label for label in node_data.get('_labels', []) if label != 'Node'), None)
        node_data.pop('_labels', None)  # Clean up the data dict
        if not node_type:
            return ""
        if (node_type.lower() == "postrequest" and node_data.get("measurement").lower() == 'usersub') or node_type.lower() == "outputcomponent":
            node_type='pr_usersub'
        term_to_lookup = node_type.lower()
        
        # The predefined map of high-frequency terms to specific URLs
        url_map = {
            "marker": "https://help.altair.com/hwsolvers/ms/topics/solvers/ms/xml-format_90.htm",
            "motion": "https://help.altair.com/hwsolvers/ms/topics/solvers/ms/xml-format_74.htm",
            "joint": "https://help.altair.com/hwsolvers/ms/topics/solvers/ms/xml-format_41.htm",
            "body": "https://help.altair.com/hwsolvers/ms/topics/solvers/ms/xml-format_35.htm",
            "postrequest": "https://help.altair.com/hwsolvers/ms/topics/solvers/ms/xml-format_83.htm",
            "pr_usersub": "https://help.altair.com/hwdesktop/hwx/topics/motionview/coordinate_systems_and_output_request_r.htm",
            "autotiresystem": "https://help.altair.com/hwdesktop/hwx/topics/motionview/fiala_tire_force_calculation_r.htm",
            "stateequation": "https://help.altair.com/hwsolvers/ms/topics/solvers/ms/gsesub_gsexx_gsexu_gseyx_gseyu.htm",
            "force": "https://help.altair.com/hwsolvers/ms/topics/solvers/ms/gfosub.htm",
            "statevariable": "https://help.altair.com/hwsolvers/ms/topics/solvers/ms/xml-format_94.htm"
            # Add more mappings as needed
        }

        url_to_load = url_map.get(term_to_lookup)
        if not url_to_load:
            print(f"  -> No documentation URL mapped for type '{node_type}'. Skipping enrichment.")
            return ""

        try:
            print(f"  -> Auto-researching type '{node_type}' for entity '{node_data.get('name')}'...")
            loader = WebBaseLoader([url_to_load])
            docs = loader.load()
            raw_content = "\n".join([doc.page_content for doc in docs])

            # Use a sub-LLM call to synthesize an explanation
            synthesis_prompt_template = """You are a helpful assistant. Your task is to explain a specific piece of a simulation model's data using the provided official documentation.

            **Official Documentation for a '{term}' component:**
            {documentation}

            **Data from User's Specific Component:**
            {data_context}

            **Your Task:**
            Based ONLY on the Official Documentation, provide a concise explanation of the key properties and values seen in the "Data from User's Specific Component".
            """
            
            synthesis_prompt = ChatPromptTemplate.from_template(synthesis_prompt_template)
            synthesis_chain = synthesis_prompt | self.llm | StrOutputParser()
            
            explanation = synthesis_chain.invoke({
                "documentation": raw_content,
                "data_context": str(node_data),
                "term": node_type
            })
            
            return f"\n--- Official Documentation Explanation of {node_name} and its properties ---\n{explanation}"

        except Exception as e:
            print(f"  -> ERROR during documentation enrichment for '{node_type}': {e}")
            return ""
        
    def _analyse_plan_step(self, state: GraphState) -> GraphState:
        """ Analyze each step in the plan to gather context from the KG. """
        step = state["plan"][state["plan_step"]]
        plan_step = state["plan_step"]

        node_explanations = []

        print(f"\n---NODE: ANALYZING PLAN STEP: '{plan_step}'---")
        # Use the toolbelt to query the KG
        step_kg_context, kg_nodes = self.tool_belt.query_graph_for_context(
            [],
            node_names=state["node_names"]
        )

        log_entry = f"--- INFO: Retrieved KG context for step '{step}':\n{step_kg_context}"
        for node in kg_nodes:
            # Enrich each node with documentation explanation
            doc_explanation = self._get_documentation_explanation(node)
            if doc_explanation:
                node_explanations.append(doc_explanation)
        
        print("---INFO: Summarizing findings for this step... ---")
        summarizer_node_prompt = ChatPromptTemplate.from_template(
            """You are an expert CAE analyst. Your current investigation goal is:
            '{step_goal}'
           
            You have retrieved the data for each entity from the knowledge graph and applied exhaustive explanation using the official documentation for each of the entity's property.
            '{explanations}'

            Based ONLY on the above details, what is your key finding for this step? What did you learn? State your conclusion clearly. If the data is inconclusive, say so.
            
            Finding:"""
        )
        summarizer_node_chain = summarizer_node_prompt | self.llm | StrOutputParser()
        node_data_goal_summary = summarizer_node_chain.invoke({
            "step_goal": step,
            "explanations": "\n".join(node_explanations)
        })

        summarizer_node_relation_prompt = ChatPromptTemplate.from_template(
            """You are an expert CAE analyst. Your current investigation goal is:
            '{step_goal}'

            You have retrieved the following data from the knowledge graph to help you. It contains multiple entities and their relationships:
            '{context}'

            Based ONLY on the above details, what is your key finding for this step? What did you learn? State your conclusion clearly. If the data is inconclusive, say so.
            
            Finding:"""
        )
        summarizer_node_relation_prompt_chain = summarizer_node_relation_prompt | self.llm | StrOutputParser()
        node_relation_goal_summary = summarizer_node_relation_prompt_chain.invoke({
            "context": "\n".join(step_kg_context),
            "step_goal": step
        })

        summarizer_node_prompt = ChatPromptTemplate.from_template(
            """You are an expert CAE analyst. Your current investigation goal is:
            '{step_goal}'
           
            You have retrieved the data for each entity from the knowledge graph and applied exhaustive explanation using the official documentation for each of the entity's property.
            You have analyzed the data of each entity individually and understood their properties in depth with respect to the current goal. Its explanation is as  follows:
            {node_data_goal_summary}
            You have also analyzed the relationships between these entities and their connections with respect to the current goal. Its explanation is as follows:
            '{node_relation_goal_summary}'
            Based ONLY on the above details, what is your key finding for this step? What did you learn? State your conclusion clearly. If the data is inconclusive, say so.
            Provide a summary of everything. Retain all numerical data. 
            
            Finding:"""
        )
        summarizer_node_chain = summarizer_node_prompt | self.llm | StrOutputParser()
        step_summary = summarizer_node_chain.invoke({
            "step_goal": step,
            "node_data_goal_summary": node_data_goal_summary,
            "node_relation_goal_summary": node_relation_goal_summary
        })

        log_entry = (
            f"Plan {state['plan_step']}: {step}\n"                
            f"  - Finding: {step_summary}"
        )
        # state["plan_step"] += 1
        state["reasoning_log"].append(log_entry)
        state["step_summaries"].append(step_summary)
        state['node_names'] = []
        state['node_types'] = []
        return state   
    
    def _generate_final_answer(self, state: GraphState) -> Generator[dict, None, None]:
        """ Generate the final answer by synthesizing all step-by-step findings. """
        print("\n---NODE: SYNTHESIZING FINAL ANSWER FROM STEP SUMMARIES---")
        
        question = state["question"]
        hypotheses = "\n".join([f"- {h}" for h in state["hypotheses"]])
        investigation_summary = "\n".join(
            [f"Step {i+1} Finding: {summary}" for i, summary in enumerate(state["step_summaries"])]
        )

        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", 
            """You are a Master CAE Expert. Your task is to provide a final root cause analysis.
            You have been given the user's original question, your initial hypotheses, and a summary of findings from your step-by-step investigation.

            Your job is to synthesize these findings into a final, conclusive answer.
            1. Review the initial hypotheses and action plan.
            2. Analyze the summarized findings from each investigation step.            
            3. Clearly state the likely root cause and explain your reasoning by referencing the step findings.
            4. Suggest concrete next steps for the user.
            """),
            ("human", 
            """
            **Original Question:**
            {question}

            **Initial Hypotheses:**
            {hypotheses}

            **Summary of Investigation Findings:**
            {investigation_summary}

            ---
            Based on your investigation, please provide the final analysis and conclusion.
            """),
        ])
        
        synthesis_chain = synthesis_prompt | self.llm | StrOutputParser()
        
        final_ans = ""
        for chunk in synthesis_chain.stream({
            "question": question,
            "hypotheses": hypotheses,
            "investigation_summary": investigation_summary
        }):
            final_ans += chunk
            yield {"answer": final_ans}
        
        return {"answer": final_ans}

    
    def _create_graph(self) -> StateGraph:
        """Creates and returns the StateGraph for the analyst agent."""
        workflow = StateGraph(GraphState)
        workflow.add_node("create_initial_hypotheses", self._create_initial_hypotheses)
        workflow.add_node("get_step_kg_context", self._get_step_kg_context)
        workflow.add_node("generate_answer", self._generate_final_answer)
        workflow.add_node("analyse_plan_step", self._analyse_plan_step)
        workflow.add_node("rewrite_plan", self._rewrite_hypothesis_plan)
        workflow.set_entry_point("create_initial_hypotheses")
        workflow.add_edge("create_initial_hypotheses", "get_step_kg_context")
        workflow.add_edge("get_step_kg_context","analyse_plan_step")
        workflow.add_edge("analyse_plan_step", "rewrite_plan")
        workflow.add_conditional_edges(
            "rewrite_plan",
            lambda state: bool(len(state["plan"])),
            {
                True: "get_step_kg_context",
                False: "generate_answer",
            }
        )
        workflow.add_edge("generate_answer", END)
        return workflow.compile()
    
    def process_message(self, message: str, chat_history: list):
        # Implement the logic to process the message using the vector store
        # and return a stream of response chunks.
        input = {
            "question": message,
            "chat_history": chat_history,
            "message": '',
            "step_summaries": [],
            "step_decisions": []
            }
        last_log_len = 0
        final_answer_started = False

        # The stream method on a compiled graph yields the state updates from each node.
        # We need to filter for the updates from our 'generate' node to get the tokens.
        for update in self.qa_graph.stream(input):
            if "reasoning_log" in update[list(update.keys())[0]]:
                current_log = update[list(update.keys())[0]]["reasoning_log"]
                if len(current_log) > last_log_len:
                    new_entry = current_log[-1]
                    yield f"STATUS: {new_entry}\n" # Yield a special token your UI can parse
                    last_log_len = len(current_log)
                    
            if "generate_answer" in update:
                if not final_answer_started:
                    yield "FINAL_ANSWER_START\n" # A token for the UI
                final_answer_started = True

                yield update["generate_answer"]["answer"]
        
        # yield f"Processed message: {message}"
    def save_graph(self, filepath: Path):
        """Saves the graph structure to a file."""
        graph = self.qa_graph.get_graph()
        try:
            # Draw the graph and save it as a PNG file
            # You can also use .draw_svg() or .draw_mermaid() for other formats
            image_data = graph.draw_mermaid_png()
            
            # Save the image data to a file
            with open(filepath, "wb") as f:
                f.write(image_data)

            print(f"✅ Graph visualization saved to {filepath}")
        except Exception as e:
            print(f"❌ Could not visualize graph. Make sure you have installed graphviz.")
            print(f"   Error: {e}")
    
qa_analyst_agent = QAAnalystAgent()
qa_analyst_agent.save_graph(Path(__file__).parent / "qa_analyst_agent_graph_2.png")