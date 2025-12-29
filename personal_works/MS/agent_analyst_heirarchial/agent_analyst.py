from pathlib import Path
from typing import List, TypedDict, Generator, Optional, Any
from operator import itemgetter
import re
from urllib import response
import base64

from neo4j_kg_builder import Neo4jConnector
from agent_tools_2 import ToolBelt
import global_vars
from global_vars import GraphState
from action_step_executor_analyst import ActionStepExecutorAnalyst

from langchain_core.messages import BaseMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.documents import Document
from pydantic.v1 import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver




class QAAnalystAgent:
    def __init__(self):
        self.EXPERT_MINDSET_TEXT = """
        Bodies are connected to other bodies through Joints. 
        Motions are applied to Joints to drive the system, 
        or can come from direct Initial Conditions on Bodies. 
        Initial conditions can be given directly on the Body properties or 
        through user-controlled subroutines like SETIC/SETWIC. 
        Motions result in changes to the position, velocity, and acceleration of the Bodies in translation and rotation.
        State variables track these changes over time by measuring them using markers attached to bodies. 
        These are used as inputs by state equations to obtain the kinematics of the current state.
        State equations perform calculations to output forces and moments that act on bodies, joints, or other entities in the system as per their connections.                
        These Forces and moments can be calulated by simple expressions or as constants or calculated via user subroutines employing state equations. 
        These forces, in turn, affect the Bodies' motion, creating a feedback loop that evolves over time. 
        PostRequests provide a way to measure and plot these physical quantities for analysis by user. Post requests measure displacements, velocities, accelerations, forces, moments, and other derived quantities at specified locations using markers attached to bodies in the model during the simulation.
        General strategy to approach any investigation
        1. Understand all the entities first from their properties as defined in the model coupled with explanations from official documentation. 
        2. Understand how the requested entity interacts with others through the details. Try to form a mental model of the entity and its relationships.
        3. Use PostRequests to understand the physical quantities of the entities. 
        4. Use this understanding to trace root causes, relationships, and dependencies in the model
        5. No need to analyse any property files at all. Just focus on the model structure and its data.        
        Whenever a general symptom is reported, if nothing else works, postrequests are often the best starting point to trace back to root causes. Identify post requests relevant to the symptom, then trace their dependencies back through forces, motions, joints, and bodies to find potential issues.
        """
        self.llm = ChatOpenAI(
            model_name=global_vars.model_openai_4omini,
            #openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            #openai_api_key=global_vars.google_api_key,
            temperature=0.3,
            streaming=True,
        )
        self.neo4j_connector = Neo4jConnector(
            uri=global_vars.NEO4J_URI,
            user=global_vars.NEO4J_USER,
            password=global_vars.NEO4J_PASSWORD
        )
        self.tool_belt = ToolBelt(self.neo4j_connector)
        self.hypotheses_revision_map =[{}]
        self.action_plan_revision_map =[{}]
        self.subgraph_executor = ActionStepExecutorAnalyst(
            tool_belt=self.tool_belt,
            llm=self.llm,
            neo4j_connector=self.neo4j_connector
        )._create_sub_graph()
        self.qa_graph = self._create_graph()

    def _create_hypotheses(self, state: GraphState) -> GraphState:
        """Create initial hypotheses based on the user's question."""
        if state.get("chat_history", None):            
            user_feedback = state["question"]
            hypotheses_prev = ""
            for entry in self.hypotheses_revision_map[1:]:
                user_question, hypothesis = list(entry.items())[0]
                hypotheses_prev += f"\nUser Question/Feedback: {user_question}\nPrevious Hypotheses:\n"
                for h in hypothesis:
                    hypotheses_prev += f"- {h}\n"            
        else:
            # feedback_llm_processed = ""
            user_feedback = ""
            hypotheses_prev = ""
        question = state["original_question"]
        # get a complete string of all the key and value pairs from the previously created hypotheses and their corresponding user feedbacks
        
        motion_solve_graph_schema = self.neo4j_connector.get_complete_schema_definition()
        
        supervisor_prompt_template = """You are a Master CAE Diagnostician for Altair MotionSolve for multibody dynamics. Your role is to act as a augmented Engineering Analyst for the human, creating a strategic plan to investigate a user's query.
        **Expert Mindset (First Principles of the System):**        
        {expert_mindset}

        **Your Goal:**
        Your goal is to use this mindset to understand the user's question and create or revise a hypotheses to investigate it.

        **Your Task (Think Step-by-Step):**
        1. Read the "Expert Mindset" to understand the various factors at play and how to approach the user's question.
        2. Read the user's question carefully.
        3. Read your previous hypotheses if any.
        4. Read the user's feedback on any previous hypotheses if provided. User's feedback may include clarifications, additional context, or corrections to your previous hypotheses. Use this to refine your understanding of the problem..
        5. Review the system schema to understand the components and relationships in the model.
        6. Based on this understanding, generate a ranked list of 2-3 distinct, testable hypotheses that could explain the user's query.
        
        **System Schema:** Available Node and Relationship types in the knowledge graph. This tells you what kinds of components and interactions exist in the model.
        {schema}
        
        **User's Orginal Question:**
        "{question}"
        **Your Previous Hypotheses if any:**
        "{hypotheses_prev}"
        **User's latest feedback on the previous hypotheses if any.**
        "{user_feedback}"

        **Formulate Hypotheses:**
        Each hypothesis should be a clear, concise statement that proposes a potential explanation for the user's question.
        Example 1: "Hypothesis: Asymmetric tire forces are causing the vehicle to pull left."
        Example 2: "Hypothesis: The hub is a element which is connected to wheel part. The forces experienced by it are based on its connections with other entities. There may or may not be any problem with it. It can be confirmed based on further analyses of output data and connected elements"
        **Your Output (respond ONLY with a valid JSON object):**
        {{
            "hypotheses": [
                "Rank 1: [Your primary hypothesis here]",
                "Rank 2: [Your secondary hypothesis here]",
                "Rank 3: [Your tertiary hypothesis here]"
            ],            
        }}
        """
        supervisor_prompt = ChatPromptTemplate.from_template(supervisor_prompt_template)
        supervisor_chain = supervisor_prompt | self.llm | JsonOutputParser()
        response = supervisor_chain.invoke({
            "expert_mindset": self.EXPERT_MINDSET_TEXT,
            "schema": motion_solve_graph_schema,
            "question": question,
            "user_feedback": user_feedback,
            # "feedback_llm": feedback_llm_processed,
            "hypotheses_prev": hypotheses_prev,
        })

        hypotheses = response["hypotheses"]
        reasoning_log = "Supervisor Analysis Complete.\n"
        reasoning_log += "\n".join([f"- {h}" for h in hypotheses]) + "\n"

        state["hypotheses"] = hypotheses
        state["reasoning_log"] = reasoning_log
        self.hypotheses_revision_map.append({user_feedback or question: hypotheses})
        return state
    
    def _intent_analysis(self, data, question) -> dict:
        """ Reformulate question based on chat history. """        
        
        contextualize_q_system_prompt = """You are an expert intention analyst. 
        Your job is to analyze the user's message and infer their intention based on the provided data.

        INTENTIONS:
        1. "Revise": The user wants to change, edit, or improve the selected items.
        2. "Confirmed": The user agrees with, selects, or wants to proceed with specific items.

        DATA FORMAT:
        The data consists of a list of hypotheses or items, usually labeled (e.g., "Rank 1", "Rank 2" or "1.", "2.").

        TASK:
        - Identify the intention.
        - If the user refers to specific items (by rank, number, or description), extract the FULL TEXT of those items from the data and place them in the "selections" list.
        
        EXAMPLE:
        Data: "- Rank 1: Gravity is 9.8\\n- Rank 2: Water boils at 100"
        User: "I like the first one"
        Output: {{"intention": "Confirmed", "selections": ["Gravity is 9.8"]}}

        The user's message is: "{question}"
        Given the following data:
        {data}

        **Your Output (respond ONLY with a valid JSON object):**
        {{
            "intention": "Revise or Confirmed",
            "selections": ["Full text of item 1", "Full text of item 2"]
        }}
        """
        
        # Using the updated prompt
        contextualize_q_prompt = ChatPromptTemplate.from_template(contextualize_q_system_prompt)
        history_retriever = contextualize_q_prompt | self.llm | JsonOutputParser()
        
        try:
            retrieved_context = history_retriever.invoke({"data": data, "question": question})
        except Exception as e:
            # Fallback in case of parsing errors
            return {"intention": "Confirmed", "selections": []}
        
        intention = retrieved_context.get("intention", "")
        selections = retrieved_context.get("selections", [])

        return {"intention": intention, "selections": selections}
    
    def _generate_investigation_plan(self, state: GraphState) -> dict:
        """ Generate an initial investigation plan based on hypotheses. """
        question = state["original_question"]
        # get a complete string of all the key and value pairs from the previously created hypotheses and their corresponding user feedbacks
        
        motion_solve_graph_schema = self.neo4j_connector.get_complete_schema_definition()
        hypotheses = state["hypotheses"]
        user_feedback = state["question"]
        if self.action_plan_revision_map and len(self.action_plan_revision_map)>1:            
            user_feedback = state["question"]
            action_plan_prev = ""
            for entry in self.action_plan_revision_map[1:]:
                user_question, action_plan = list(entry.items())[0]
                action_plan_prev += f"\nUser Question/Feedback: {user_question}\nPrevious Action Plan:\n"
                for h in action_plan:
                    action_plan_prev += f"- {h}\n"
        else:
            action_plan_prev = ""
            user_feedback = ""

        supervisor_prompt_template = """You are a Master CAE Diagnostician for Altair MotionSolve for multibody dynamics. Your role is to act as a augmented Engineering Analyst for the human, creating a strategic plan to investigate a user's query.
        **Expert Mindset (First Principles of the System):**        
        {expert_mindset}

        **Your Goal:**
        Your goal is to use this mindset to understand the user's question and create or revise a step by step action plan based on a earlier agreed upon hypothesis. 
        You will have access to the original model, the initial data defined in the entities and the relationships of various entities in the model through neo4j knowledge graphtime
        You will have access to time series data from various output requests providing physical meaning of the simulation results.
        You will have access to write python code to analyze the time series data to find patterns, correlations, and insights.
        You will have access to write cypher queries to extract specific information from the knowledge graph.

        **Your Task (Think Step-by-Step):**
        1. Read the "Expert Mindset" to understand the various factors at play and how to approach the user's question and the hypothesis.
        2. Read the user's question carefully.
        3. Read the hypotheses that was discussed and agreed with the user.        
        5. Review the system schema to understand the components and relationships in the model.
        6. Based on this understanding, define clear, concrete step by step actions fulfill the hypothesis.
        7. Each step should specify what is being analyses, how its being analyzed, what is expected and what data or tools to use.
        8. Ensure the steps are logically ordered to build upon each other.
        9. Understand the entities by collecting the necessary data of each entity. They will be enriched with explanations from the official documentation.
        10. Understand how the requested entity interacts with others through the details. Try to form a mental model of the entity and its relationships. Request new entities accordingly.
        11. Use PostRequests time series data to understand the variations in physical quantities of these entities.
        12. Use this understanding to trace root causes, relationships, and dependencies in the model
        14. User may provide feedback on the plan. Use this to refine and improve the plan.
        
        After the plan has been agreed upon, and as it gets executed using real data, you will get new insights which will lead you to most probably revise the plan. So no need to delve very deeply into all steps initially. Provide the first couple of steps with high confidence.
        
        **System Schema:** Available Node and Relationship types in the knowledge graph. This tells you what kinds of components and interactions exist in the model.
        {schema}
        
        **User's Orginal Question:**
        "{question}"
        **Agreed upon Hypotheses:**
        "{hypotheses}"
        **User's latest feedback on the previous action plan if any.**
        "{user_feedback}"
        **Previous Action Plan if any:**
        "{action_plan_prev}"
        **Formulate/revise Action Plan:**
        **Your Output (respond ONLY with a valid JSON object):**
        {{
            "action_plan": "[
                1. Your clear, actionable step 1 here.",
                2. Your clear, actionable step 2 here.
                3. ...
                list all the steps needed to investigate the hypothesis
            ]"
        }}
        """
        supervisor_prompt = ChatPromptTemplate.from_template(supervisor_prompt_template)
        supervisor_chain = supervisor_prompt | self.llm | JsonOutputParser()
        response = supervisor_chain.invoke({
            "expert_mindset": self.EXPERT_MINDSET_TEXT,
            "schema": motion_solve_graph_schema,
            "question": question,
            "user_feedback": user_feedback,
            # "feedback_llm": feedback_llm_processed,
            "action_plan_prev": action_plan_prev,
            "hypotheses": "\n".join(hypotheses),
        })

        action_plan = response["action_plan"]
        reasoning_log = "Supervisor Analysis Complete on Action Plan.\n"
        reasoning_log += "\n".join([f"- {ap}" for ap in action_plan]) + "\n"

        state["action_plan"] = action_plan
        state["reasoning_log"] = reasoning_log
        self.action_plan_revision_map.append({user_feedback or question: action_plan})
        return state

    def _execute_action_steps(self, state: GraphState) -> dict:
        """ Execute a single investigation step using tools. """
        state["current_step"] = state.get("action_plan", [])[0] if state.get("action_plan") else None
        return state  # Implementation would go here

    def _generate_final_answer(self, state: GraphState) -> Generator[dict, None, None]:
        """ Generate the final answer by synthesizing all step-by-step findings. """
        print("\n---NODE: SYNTHESIZING FINAL ANSWER FROM STEP SUMMARIES---")
        
        question = state["original_question"]
        hypotheses = "\n".join([f"- {h}" for h in state["hypotheses"]])
        action_plan = "\n".join([f"- {step}" for step in state.get("action_plan", [])])
        step_summaries = "\n".join([f"- {summary}" for summary in state.get("step_summaries", [])])
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", 
            """You are a Master CAE Expert. Your task is to assist the human Engineer in his analysis. The user starts with a message, dicusses and agrees upon initial hypotheses, then discusses and agrees upon an action plan to investigate those hypotheses step-by-step. You can be in any one of the following situations:
            Your task. It should be one of the following based on the context:
            1. Display the current hypotheses as long as the state is still in "hypothesis" phase.
            2. Display the current action plan as long as the state is still in "planning" phase            
            3. Start analysing the step summaries as soon as phase enters "execution"
               Your task is to analyze the step summaries and provide a analysis related to the original question.
            
            When executing step 3, follow these guidelines:
            1. Review the initial hypotheses 
            2. Review the action plan.
            3. Analyze the summarized findings from each investigation step. 
            4. Clearly state your analysis and explain your reasoning by referencing the step findings.
            5. Suggest concrete next steps for the user.
            """),
            ("human", 
            """
            **Original Question:**
            {question}

            **Initial Hypotheses:**
            {hypotheses}

            **Action Plan:** 
            {action_plan}
            **Findings from Investigation Steps:**
            {step_summaries}            
            ---
            Current phase is {current_phase}. Based on your investigation, please provide the final analysis and conclusion.
            """),
        ])
        
        synthesis_chain = synthesis_prompt | self.llm | StrOutputParser()
        
        final_ans = ""
        for chunk in synthesis_chain.stream({
                                                "question": question,
                                                "hypotheses": hypotheses,
                                                "action_plan": action_plan,
                                                "step_summaries": step_summaries,
                                                "current_phase": state["current_phase"],
                                            }):
            final_ans += chunk
            yield {"answer": final_ans}
        
        return {"answer": final_ans}
    
    def _user_intention_understanding(self, state: GraphState) -> dict:
        """ Check if the user intends to create hypotheses or action plan or execute steps. """
        # the llm has to find out from the question what to do next
        # this can be based on keywords or intent classification
        # for simplicity, we use keyword matching here
        question = state["question"].lower()
        
        if not state.get("chat_history",None):
            state["original_question"]=question
            state["current_phase"] = "hypothesis"
        if state["current_phase"] == "hypothesis":
            hypotheses_prev = ""
            user_feedback = state["question"]
            if self.hypotheses_revision_map and len(self.hypotheses_revision_map)>1:
                entry = self.hypotheses_revision_map[-1]
                user_question, hypothesis = list(entry.items())[0]
                hypotheses_prev += f"\nAI response:\n"
                for h in hypothesis:
                    hypotheses_prev += f"- {h}\n"
                # hypotheses_prev += f"User Question/Feedback: {user_question}\n"
                feedback_llm_processed = self._intent_analysis(hypotheses_prev, user_feedback)
                if feedback_llm_processed['intention'] == "Confirmed":
                    state["hypotheses"] = feedback_llm_processed['selections']
                    state['reasoning_log'] = "User confirmed the hypotheses:\n" 
                    state['current_phase'] = "planning"
            
        
        elif state["current_phase"] == "planning":
            action_plan_prev = ""
            user_feedback = state["question"]
            if self.action_plan_revision_map and len(self.action_plan_revision_map)>1:
                entry = self.action_plan_revision_map[-1]
                user_question, action_plan = list(entry.items())[0]
                action_plan_prev += f"Previous Action Plan by AI:\n"
                for ap in action_plan:
                    action_plan_prev += f"- {ap}\n"
                feedback_llm_processed = self._intent_analysis(action_plan_prev, user_feedback)
                if feedback_llm_processed['intention'] == "Confirmed":
                    state["action_plan"] = feedback_llm_processed['selections']
                    state['reasoning_log'] = "User confirmed the action plan:\n" 
                    state['current_phase'] = "execution"
            
        elif state['current_phase'] == "execution":
            user_feedback = state["question"]

            state["current_phase"] = "execution"
        else:
            state["current_phase"] = "finalizing"
        return state

    def _router(self, state: GraphState) -> str:
        """Decide the next node to execute based on the current state."""
        return state
        
    def _create_graph(self) -> StateGraph:
        workflow = StateGraph(GraphState)
        # checkpointer = MemorySaver()
        workflow.add_node("router", self._router)
        workflow.add_node("user_intention_understanding", self._user_intention_understanding)
        workflow.add_node("generate_investigation_plan", self._generate_investigation_plan)
        workflow.add_node("execute_action_steps", self._execute_action_steps)
        workflow.add_node("execute_subgraph_step", self.subgraph_executor)
        # workflow.add_node("history_aware_retrieval", self._history_aware_retrieval)
        workflow.add_node("create_hypotheses", self._create_hypotheses)
        workflow.add_node("generate_final_answer", self._generate_final_answer)

        workflow.set_entry_point("user_intention_understanding")
        workflow.add_edge('user_intention_understanding', 'router')
        workflow.add_conditional_edges(
            "router",
            lambda state: state['current_phase'],
            {
                'hypothesis': 'create_hypotheses',
                'planning': 'generate_investigation_plan',
                'execution': 'execute_action_steps',
                'finalizing': 'generate_final_answer',
            }
        )

        # 4. All task nodes go to END (to wait for the next user message)
        workflow.add_edge("create_hypotheses", "generate_final_answer")
        workflow.add_edge("generate_investigation_plan", "generate_final_answer")
        workflow.add_edge("execute_action_steps", "execute_subgraph_step")
        workflow.add_edge("execute_subgraph_step", "generate_final_answer")
        workflow.add_edge("generate_final_answer", END)
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)        
        
    
    def process_message(self, message: str, chat_history: List[BaseMessage]) -> Generator[str, None, None]:
        inputs = {
            "question": message,
            "chat_history": chat_history,
        }
        config = {"configurable": {"thread_id": "user_session_123"}}
        final_answer_started = False

        for update in self.qa_graph.stream(inputs, config=config):
            if "reasoning_log" in update[list(update.keys())[0]]:
                current_log = update[list(update.keys())[0]]["reasoning_log"]
                new_entries = current_log
                yield f"STATUS: {new_entries}\n"
            if "generate_final_answer"in update:
                final_answer = update["generate_final_answer"].get("answer", "")
                if final_answer and not final_answer_started:
                    final_answer_started = True
                    yield "FINAL_ANSWER_START\n"
                if final_answer:
                    yield final_answer

    def save_graph(self, filepath: Path):
            import requests
            """Saves the graph structure to a file."""
            graph = self.qa_graph.get_graph()
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
    

qa_analyst_agent = QAAnalystAgent()
qa_analyst_agent.save_graph(Path(__file__).parent / "qa_analyst_agent_h_graph.png")        