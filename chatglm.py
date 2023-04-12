#envs pytorch2
#pip install protobuf==3.20.0 transformers==4.27.1 icetk cpm_kernels
#pip install -r requirements.txt

proxies = {
    'http://': 'http://127.0.0.1:7890', 
    'https://': 'http://127.0.0.1:7890'
}

import gradio as gr
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("/home/lyx22/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/home/lyx22/chatglm-6b", trust_remote_code=True).quantize(4).half().cuda()
model = model.eval()



def ask_chatglm(text):
    prompt = f'对下面这段文本进行关系抽取：“{text}”。要求为提取其中的实体名称，并确定它们之间的关系。注意你只需要回答关系，回答的模式为“实体1-关系-实体2”，在一行内输出一条关系，并且不用注明“实体1”、“关系”、“实体2”等'
    response,_ = model.chat(tokenizer, prompt, history=[])
    return response.strip()


def re_extrac(text: str):
    text = text.strip()
    re = ask_chatglm(text=text)
    return re


gr.close_all()
demo = gr.Interface(fn=re_extrac, inputs=gr.Textbox(lines=3, placeholder='请在此处输入文本'), outputs="text",
                    examples=[['史蒂芬·斯皮尔伯格是美国著名电影导演，曾三次获得奥斯卡，因为执导了辛德勒的名单，拯救大兵瑞恩和斯坦利·库布里克的奇幻之旅'],
                              ['周华健演唱的刀剑如梦是一首非常好听的歌']],
                    title='基于chatlm-6b关系抽取',
                    description='在"text"框输入待分析段落,自动抽取文中关系')

demo.launch()