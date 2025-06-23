import os
system_gpt_msg = "you are a helpful assistant. Reply in markdown. You're not just a command-taker; you're a helpful assistant. If you think a picture would make your answer better, you have a tool for that. Use it"
system_gemini_msg = "You are a funny helpful assistant. Reply in markdown."
prompt_gpt=[
    {"role":"system", "content":system_gpt_msg}
]


openai_api_key=os.getenv('OPEN_API_KEY')
google_api_key=os.getenv('GOOGLE_API_KEY')
model_openai_4_5_preview = 'gpt-4.5-preview'
model_openai_4omini = 'gpt-4o-mini'
model_openai_4onano = 'gpt-4.1-nano'
model_openai_o3mini = 'o3-mini'
model_genimi = 'gemini-2.0-flash'
model_gemini_2_5_flash_4_17 = "gemini-2.5-flash-preview-04-17"
model_gemini_2_5_flash_5_20 = "gemini-2.5-flash-preview-05-20"
model_gemini_2_5_pro_preview='gemini-2.5-pro-preview-06-05'

gpt_caller=None
gemini_caller=None
claude_caller=None