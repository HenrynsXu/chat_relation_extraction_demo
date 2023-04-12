import gradio as gr
import openai
import requests,json

proxies = {
    'http://': 'http://127.0.0.1:7890', 
    'https://': 'http://127.0.0.1:7890'
}

def ask_GPT(text):
    
    openai.api_key = 'YOUR_API_KEY'
    prompt = f'对下面这段文本进行关系抽取：“{text}”。要求为提取其中的实体名称，并确定它们之间的关系。注意你只需要回答关系，回答的模式为“实体1-关系-实体2”，在一行内输出一条关系，并且不用注明“实体1”、“关系”、“实体2”等'
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = [{"role":"user","content":prompt}],
            temperature=0.5)
    ans=response['choices'][0]['message']['content']
    return ans.strip()

def re_extrac(text:str):
    text = text.strip()
    re = ask_GPT(text=text)
    return re
    


demo = gr.Interface(fn=re_extrac, inputs=gr.Textbox(lines=3,placeholder='请在此处输入文本'), outputs="text",
                    examples=[['史蒂芬·斯皮尔伯格是美国著名电影导演，曾三次获得奥斯卡，因为执导了辛德勒的名单，拯救大兵瑞恩和斯坦利·库布里克的奇幻之旅']],
                    title='基于GPT-3.5关系抽取',
                    description='在"text"框输入待分析段落,自动抽取文中关系')

demo.launch(server_port=41321,server_name='0.0.0.0') 
