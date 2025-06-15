system_gpt_msg = "you are a helpful assistant. Reply in markdown"
system_gemini_msg = "You are a funny helpful assistant. Reply in markdown."
prompt_gpt=[
    {"role":"system", "content":system_gpt_msg}
]
from openai import OpenAI
import google.generativeai,os
