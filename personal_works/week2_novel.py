#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display
from pathlib import Path
import nbimporter


# In[2]:


import utility_fncs
from global_vars import *
import global_vars

# In[3]:


from openai import OpenAI
import google.generativeai


# In[4]:


load_dotenv(override=True)



# In[5]:


openai = OpenAI()
google.generativeai.configure(api_key=google_api_key)


# In[6]:


#gemini_via_openai_client = OpenAI(
#    api_key=google_api_key,
#    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
#)
#response = gemini_via_openai_client.chat.completions.create(
#    model=model_gemini_2_5_flash_4_17,
#    messages=prompts,
#)
#print(response.choices[0].message.content)


# In[7]:


def call_gpt(msg, filedata=None):
    #print(filedata)   
    prompt_gpt.append({"role": "user", "content":msg})
    if filedata: prompt_gpt.append({"role": "user", "content":utility_fncs.getFileContent(filedata)})
    completion = openai.chat.completions.create(
        model=model_openai_4onano,
        messages=prompt_gpt,
        stream=True,
    )
    response=""
    for chunk in completion:
        response+= chunk.choices[0].delta.content or ''
        yield response
    #reply=completion.choices[0].message.content
    prompt_gpt.append({"role":"assistant", "content":response})
    



# In[8]:


prompt_google_gemini = []
gemini = google.generativeai.GenerativeModel(
        model_name = model_gemini_2_5_flash_5_20,    
        system_instruction=system_gemini_msg
    )

def upload_file_gemini(file_path):
    file = google.generativeai.upload_file(file_path)
    #print(file.uri)
    return file

def call_google_gemini(msg, file_path=None):
    prompt_msg=[{"text": msg}]

    
    prompt_google_gemini.append({"role": "user", "parts": prompt_msg})
    completion = gemini.generate_content(prompt_google_gemini,stream=True)
    
    response = ""
    for chunk in completion:
        response+= chunk.text or ''
        yield response
    prompt_google_gemini.append({"role":"model","parts":[{"text":response}]})
    
    


# In[9]:


def start_conv():
    print(f"Gemini: Hi\n")
    reply_google='Hi'
    for i in range(5):
        reply_gpt = call_gpt(reply_google)
        print(f"GPT: {reply_gpt}\n")
        reply_google = call_google_gemini(reply_gpt)
        print(f"Google: {reply_google}\n")



# In[10]:
def set_gpt_caller(func):    
    global_vars.gpt_caller = func    
    
def set_gemini_caller(func):
    global_vars.gemini_caller = func
    
    
def dispatch_ai_call(qt,file,model):
    
    if model=='GPT':
        #yield from
        return global_vars.gpt_caller(qt, file)
    elif model=='GEMINI':
        #yield from
        return global_vars.gemini_caller(qt,file)



# In[11]:


#dispatch_ai_call("how are you. can you help me with this file","C:/Projects/llm_engg/llm_engineering/requirements.txt",'GEMINI')


# In[ ]:




