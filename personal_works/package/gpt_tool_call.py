# openai_agent.py

import base64
import json, os
from io import BytesIO
from typing import Literal

from openai import OpenAI
from PIL import Image
from typing import Callable

# Import the utility class from the other file
from app_utils import AppUtils
import global_vars
from dotenv import load_dotenv
import independent_tool_functions

class OpenAIAgent:
    """
    An OpenAI chat agent capable of using external tools to answer questions
    and perform tasks like generating images.
    """

    def __init__(self, model: str = global_vars.model_openai_4omini, system_message: str = None):
        """
        Initializes the OpenAIAgent.

        Args:
            model (str): The OpenAI model to use for chat completions.
            system_message (str, optional): An initial system prompt.
        """
        
        
        self.client = OpenAI()
        self.model = model
        self.conversation_history = []
        self.available_tools = {}
        if system_message:
            self.conversation_history.append({"role": "system", "content": system_message})

        # Define instance methods that will be used as tools
        #self.available_tools = {
        #    "get_current_weather": self.get_current_weather,
        #    "get_stock_price": self.get_stock_price,
        #    "create_image": self.create_image,
        #}
        
        # Generate tool schemas using the AppUtils class
        self.tools = [
            AppUtils.create_tool_from_function(func) 
            for func in self.available_tools.values()
        ]

    def add_tool(self, func: Callable):
        """
        public facing function to add a callable python function as a tool for openAI

        Args:
            func (callable): The user's callable function

        Returns:
            None: Adds the function to the class's list of available tool
        """
    
        self.available_tools[func.__name__]=func
        
        for f in self.tools:
            if func.__name__ == f.get('function').get("name"):
                return
        
        self.tools.append(AppUtils.create_tool_from_function(func)) 
    
    # --- Core Chat Logic ---
    
    def reset_history(self, system_message: str = None):
        """Resets the conversation history, optionally with a new system message."""
        self.conversation_history = []
        if system_message:
            self.conversation_history.append({"role": "system", "content": system_message})

    def chat(self, user_prompt: str):
        """
        Handles a user's chat prompt, calls tools if necessary, and returns a response.

        Args:
            user_prompt (str): The user's input message.

        Returns:
            tuple: A tuple containing the final text response (str) and any generated image (PIL.Image or None).
        """
        self.conversation_history.append({"role": "user", "content": user_prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            tools=self.tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        self.conversation_history.append(response_message)
        
        tool_calls = response_message.tool_calls
        generated_image = None

        if not tool_calls:
            return response_message.content, None

        # --- Handle Tool Calls ---
        tool_messages = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = self.available_tools.get(function_name)
            
            if not function_to_call:
                print(f"Error: Model tried to call unknown function '{function_name}'")
                continue
            
            try:
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(**function_args)
                
                # Unpack the response from our tool functions
                if function_response.get('type') == 'image':
                    generated_image = function_response.get('content')
                    response_content = "Image has been successfully created."
                else:
                    response_content = function_response.get('content', '')

                tool_messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": response_content,
                })
                
            except Exception as e:
                print(f"Error calling function {function_name}: {e}")
        
            self.conversation_history.append(*tool_messages)
        
        # Second API call to get a natural language summary
        final_response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
        )
        final_message = final_response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": final_message})
        
        return final_message, generated_image


if __name__ == '__main__':
    # --- Example Usage ---
    
    # Initialize the agent
    agent = OpenAIAgent(system_message="You are a helpful assistant.")
    agent.add_tool(independent_tool_functions.get_current_weather)
    agent.add_tool(independent_tool_functions.create_image)
    # --- Run a Gradio UI ---
    # The `chat` method is directly compatible with the Gradio launchers.
    # Note: To run this, you need to install gradio: pip install gradio
    #
    # # Create a wrapper to fit the Gradio input/output format
    def gradio_interface(prompt, file, model_selection):
        # This is a simplified interface; file and model are not used in this example
        # but are there to match the launcher's expected inputs.
        text_response, image_response = agent.chat(prompt)
        return text_response, image_response

    # Use the utility to create the launcher
    launcher = AppUtils.get_gradio_multi_modal_launcher(gradio_interface)
    
    print("Launching Gradio interface... Go to the provided URL in your browser.")
    # The Gradio app will be available at a local URL like http://127.0.0.1:7860
    launcher.launch()
    del agent