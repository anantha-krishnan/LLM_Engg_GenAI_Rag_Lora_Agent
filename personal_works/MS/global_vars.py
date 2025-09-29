import os
#import vertexai
#from vertexai.preview.vision_models import ImageGenerationModel, Image
from IPython.display import Image as IPythonImage
import anthropic
from dotenv import load_dotenv
from pathlib import Path

system_gpt_msg = "you are a helpful assistant. Reply in markdown. You're not just a command-taker; you're a helpful assistant. If you think a picture would make your answer better, you have a tool for that. Use it"
system_gemini_msg = "You are a funny helpful assistant. Reply in markdown."
prompt_gpt=[
    {"role":"system", "content":system_gpt_msg}
]
# use pathlib to get current directory
current_dir = Path(__file__).resolve().parent
# go two levels up
parent_dir = current_dir.parent.parent

# join with relative path to .env file

env_path = parent_dir / '.env'

load_dotenv(env_path,override=True)

openai_api_key=os.getenv('OPENAI_API_KEY')
google_api_key=os.getenv('GOOGLE_API_KEY')
ANTHROPIC_api_key=os.getenv('ANTHROPIC_API_KEY')
HF_token=os.getenv('HF_TOKEN')
model_openai_4_5_preview = 'gpt-4.5-preview'
model_openai_4onano = 'gpt-4.1-nano'
model_openai_4o = 'gpt-4o'
model_openai_o3mini = 'o3-mini'
model_genimi = 'gemini-2.0-flash'
model_gemini_2_5_flash_4_17 = "gemini-2.5-flash-preview-04-17"
model_gemini_2_5_flash_5_20 = "gemini-2.5-flash-preview-05-20"
model_gemini_2_5_pro_preview='gemini-2.5-pro-preview-06-05'
model_gemini_1_5_pro='gemini-1.5-pro'
model_gemini_2_0_flash_image='gemini-2.0-flash-preview-image-generation'
gemini_text_model = model_gemini_1_5_pro
gemini_image_model = model_gemini_2_0_flash_image
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"

# ultra-low cost models
model_openai_4omini = "gpt-4o-mini"
CLAUDE_MODEL = "claude-3-haiku-20240307"

gpt_caller=None
gemini_caller=None
claude_caller=None