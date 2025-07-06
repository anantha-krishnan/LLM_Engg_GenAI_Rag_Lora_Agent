# gemini_agent.py

from io import BytesIO
from PIL import Image as PILImage
from typing import Callable
import os
# Google GenAI (for Chat)
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from dotenv import load_dotenv

# Vertex AI (for Image Generation)
#import vertexai
#from vertexai.preview.vision_models import ImageGenerationModel

# Import the utility class from our other file
from app_utils import AppUtils
import global_vars, independent_tool_functions

class GeminiAgent:
    """
    A Google Gemini agent capable of using external tools, including
    image generation via Vertex AI's Imagen model.
    """

    def __init__(self, model_name: str=global_vars.gemini_text_model):
        """
        Initializes the GeminiAgent.

        Args:
            model_name (str): The Gemini model name (e.g., "gemini-1.5-pro-preview-0409").
            project_id (str): Your Google Cloud Project ID.
            location (str): The Google Cloud location (e.g., "us-central1").
        """
        #self.project_id = project_id
        #self.location = location
        self.conversation_history = []
        self.gemini_tool_config = []
        # Initialize Vertex AI for image generation
        #print(f"Initializing Vertex AI with Project ID: {project_id} and Location: {location}")
        #vertexai.init(project=self.project_id, location=self.location)
        #self.image_gen_model = ImageGenerationModel.from_pretrained("imagegeneration@006")

        # Define the tools available to this agent
        self.available_tools ={
            independent_tool_functions.create_image_with_text_model_gemini.__name__:independent_tool_functions.create_image_with_text_model_gemini
        }
        
        # Create Gemini-compatible tool declarations using our AppUtils adapter
        #all_declarations = [
        #    self._create_gemini_declaration(func) 
        #    for func in self.available_tools.values()
        #]
        #gemini_tool_config = Tool(function_declarations=all_declarations)
        load_dotenv('C:/Projects/llm_engg/llm_engineering/.env',override=True)
        google_api_key=os.getenv('google_api_key')
        genai.configure(api_key=google_api_key)
        self.add_tool(independent_tool_functions.create_image_with_text_model_gemini)
        # Initialize the Gemini chat model with the tools
        self.chat_model = genai.GenerativeModel(
            model_name=model_name, 
            tools=self.gemini_tool_config
        )

    
    
    def add_tool(self, func: Callable):
        """
        public facing function to add a callable python function as a tool for openAI

        Args:
            func (callable): The user's callable function

        Returns:
            None: Adds the function to the class's list of available tool
        """
        
        for f in self.gemini_tool_config:
            if func.__name__ == f.function_declarations[0].name:
                return
        
        self.gemini_tool_config.append(self._create_gemini_declaration(func)) 
        
    @staticmethod
    def _create_gemini_declaration(func) -> FunctionDeclaration:
        """
        Adapter function to convert an OpenAI tool schema (from AppUtils)
        into a Gemini FunctionDeclaration.
        """
        openai_tool_dict = AppUtils.create_tool_from_function(func)
        function_details = openai_tool_dict['function']
        return Tool(function_declarations=[FunctionDeclaration(
            name=function_details['name'],
            description=function_details['description'],
            parameters=function_details['parameters']
            )])

    # --- Tool Definitions ---

    # --- Core Chat Logic ---

    def chat(self, user_prompt: str):
        """
        Handles a user's prompt, calls tools if necessary, and returns a response.

        Args:
            user_prompt (str): The user's input message.

        Returns:
            tuple: A tuple containing the final text response (str) and any generated image (PIL.Image or None).
        """
        self.conversation_history.append({'role': 'user', 'parts': [{'text': user_prompt}]})
        
        # First API call
        response = self.chat_model.generate_content(self.conversation_history)
        response_part = response.candidates[0].content.parts[0]
        
        generated_image = None

        if hasattr(response_part, 'function_call') and response_part.function_call:
            function_call = response_part.function_call
            print(f"Gemini wants to call function: '{function_call.name}'")
            
            # Add the model's function call request to history
            self.conversation_history.append({'role': 'model', 'parts': [response_part]})

            function_to_call = self.available_tools[function_call.name]
            function_args = {key: value for key, value in function_call.args.items()}
            
            # Execute the function
            function_response_data = function_to_call(**function_args)
            
            # Process the tool's response
            if function_response_data.get('type') == 'image':
                generated_image = function_response_data.get('content')
                function_response_content = "Image created successfully as per user's request."
            else:
                function_response_content = function_response_data.get('content', '')

            # Add the tool's result to history
            tool_response_part = {
                'function_response': {
                    "name": function_call.name,
                    "response": {"result": function_response_content}
                }
            }
            self.conversation_history.append({'role': 'tool', 'parts': [tool_response_part]})
            
            # Second API call to get the final text response
            final_response = self.chat_model.generate_content(self.conversation_history)
            final_text = final_response.text
            self.conversation_history.append({'role': 'model', 'parts': [{'text': final_text}]})
            
            return final_text, generated_image
        else:
            # If no tool was called, just return the text
            final_text = response.text
            self.conversation_history.append({'role': 'model', 'parts': [{'text': final_text}]})
            return final_text, None

if __name__ == '__main__':
    # --- Example Usage ---

    # IMPORTANT: Replace with your actual Project ID.
    # You may also need to run `gcloud auth application-default login` in your terminal.
    #GCP_PROJECT_ID = "gen-lang-client-0023157158"  # <--- REPLACE WITH YOUR PROJECT ID
    #GCP_LOCATION = "us-central1"
    #MODEL_NAME = "gemini-1.5-pro-preview-0409"
    # Initialize the agent
    agent = GeminiAgent()
    agent.add_tool(independent_tool_functions.create_image_with_text_model_gemini)
        
    def gradio_interface(prompt, file, model_selection):
        text_response, image_response = agent.chat(prompt)
        return text_response, image_response
        
    # Use the utility to create the launcher
    launcher = AppUtils.get_gradio_multi_modal_launcher(gradio_interface)
    
    print("Launching Gradio interface... Go to the provided URL in your browser.")
    # The Gradio app will be available at a local URL like http://127.0.0.1:7860
    launcher.launch()
    del agent
    print('exited completely')