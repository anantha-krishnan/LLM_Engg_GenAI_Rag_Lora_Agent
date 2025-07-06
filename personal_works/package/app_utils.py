# app_utils.py

import inspect
from pathlib import Path
from typing import get_origin, get_args, Literal, Callable

import filetype
import gradio as gr


class AppUtils:
    """A collection of static utility methods for the OpenAI Agent.
    
    This class provides helpers for:
    - Generating OpenAI tool schemas from Python functions.
    - Creating Gradio UI components.
    - Handling file operations and MIME type detection.
    - Formatting API messages.
    """

    @staticmethod
    def _is_ascii(file_path: str) -> bool:
        """Checks if a file contains only ASCII characters."""
        try:
            with open(file_path, 'rb') as f:
                f.read().decode('ascii')
            return True
        except (UnicodeDecodeError, FileNotFoundError):
            return False

    @staticmethod
    def _get_file_path_str(file_path: str | Path) -> str:
        """Converts a Path object to a string if necessary."""
        if isinstance(file_path, Path):
            return file_path.as_posix()
        return file_path

    @staticmethod
    def detect_mime_type(file_path: str | Path) -> str | None:
        """Detects the MIME type of a file."""
        path_str = AppUtils._get_file_path_str(file_path)
        if AppUtils._is_ascii(path_str):
            return "text/plain"

        kind = filetype.guess(path_str)
        return kind.mime if kind else None

    @staticmethod
    def get_file_content(fp: str | Path) -> str:
        """Reads and returns the content of a text file."""
        path_str = AppUtils._get_file_path_str(fp)
        try:
            with open(path_str, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception:
            return ""

    @staticmethod
    def get_gradio_launcher(func: Callable) -> gr.Interface:
        """Creates a standard Gradio interface for a chat function."""
        return gr.Interface(
            fn=func,
            inputs=[
                gr.Textbox(label="Your message:"),
                gr.File(label="Upload file"),
                gr.Dropdown(["GPT", "GEMINI"], label="Select model")
            ],
            outputs=[gr.Markdown(label="Response:")],
            title="Chat Agent",
            allow_flagging="never"
        )

    @staticmethod
    def get_gradio_multi_modal_launcher(func: Callable) -> gr.Interface:
        """Creates a multi-modal Gradio interface for chat and image output."""
        return gr.Interface(
            fn=func,
            inputs=[
                gr.Textbox(label="Your message:"),
                gr.File(label="Upload file"),
                gr.Dropdown(["GPT", "GEMINI"], label="Select model")
            ],
            outputs=[
                gr.Markdown(label="Response:"),
                gr.Image(label="Generated Image", height=512)
            ],
            title="Multi-Modal Chat Agent",
            allow_flagging="never"
        )
        
    @staticmethod
    def create_tool_from_function(func: Callable) -> dict:
        """
        Generates an OpenAI-compatible tool definition from a Python function.

        This function leverages type hints and a structured docstring to create 
        the JSON schema required by the OpenAI API for tool-calling.
        """
        full_docstring = inspect.getdoc(func)
        if not full_docstring:
            raise ValueError("The function must have a docstring to be used as a tool.")
        
        # Split docstring into description and args
        parts = full_docstring.split("\n\nArgs:\n")
        func_description = parts[0].strip()
        
        param_docs = {}
        if len(parts) > 1:
            current_param_name = None
            for line in parts[1].split('\n'):
                line = line.strip()
                if ':' in line:
                    param_name, param_desc = line.split(':', 1)
                    param_name = param_name.split('(')[0].strip()
                    param_docs[param_name] = [param_desc.strip()]
                    current_param_name = param_name
                elif current_param_name:
                    param_docs[current_param_name].append(line)
        
        for name, desc_lines in param_docs.items():
            param_docs[name] = "\n".join(desc_lines)
            
        sig = inspect.signature(func)
        parameters_schema = {"type": "object", "properties": {}, "required": []}
        type_mapping = {str: "string", int: "integer", float: "number", bool: "boolean"}

        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            
            if param.default == inspect.Parameter.empty:
                parameters_schema["required"].append(name)

            param_info = {}
            origin_type = get_origin(param.annotation)
            
            if origin_type is Literal:
                param_info["type"] = "string"
                param_info["enum"] = list(get_args(param.annotation))
            else:
                param_info["type"] = type_mapping.get(param.annotation, "string")
            
            if name in param_docs:
                param_info["description"] = param_docs[name]
            
            parameters_schema["properties"][name] = param_info

        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func_description,
                "parameters": parameters_schema,
            },
        }