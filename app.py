import gradio as gr
import openai
import requests,json

proxies = {
    'http://': 'http://127.0.0.1:7890', 
    'https://': 'http://127.0.0.1:7890'
}

def ask_GPT(rel,text):
    
    openai.api_key = 'OPENAI_API_KEY'
    prompt = f'给定下面一段文本"{text}"，找出其中的"{rel}"关系'
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = [{"role":"user","content":prompt}],
            temperature=0.5)
    ans=response['choices'][0]['message']['content']
    return ans.strip()

def re_extrac(text:str,relation:str):
    text = text.strip()
    re = ask_GPT(rel=relation,text=text)
    return re
    


demo = gr.Interface(fn=re_extrac, inputs=["text",'text'], outputs="text",
                    examples=[['史蒂芬·斯皮尔伯格是美国著名电影导演，曾三次获得奥斯卡，因为执导了辛德勒的名单，拯救大兵瑞恩和斯坦利·库布里克的奇幻之旅','导演-作品']],
                    title='基于GPT-3.5关系抽取',
                    description='在"text"框输入待分析段落，在"relation"框输入想要抽取的关系')

demo.launch() 
