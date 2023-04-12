import gradio as gr
import openai

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("./chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("./chatglm-6b", trust_remote_code=True).quantize(4).half().cuda()
model = model.eval()

def ask_GPT(text):
    
    openai.api_key = 'API_KEY'
    prompt = f'对下面这段文本进行关系抽取：“{text}”。要求为提取其中的实体名称，并确定它们之间的关系。注意你只需要回答关系，回答的模式为“实体1-关系-实体2”，在一行内输出一条关系，并且不用注明“实体1”、“关系”、“实体2”等，你需要抽取出尽可能多的关系。'
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = [{"role":"user","content":prompt}],
            temperature=0.5)
    ans=response['choices'][0]['message']['content']
    return ans.strip()

def ask_glm(text):
    response, _ = model.chat(tokenizer, f'对下面这段文本进行关系抽取：“{text}”。要求为提取其中的实体名称，并确定它们之间的关系。注意你只需要回答关系，回答的模式为“实体1-关系-实体2”，在一行内输出一条关系，并且不用注明“实体1”、“关系”、“实体2”等，你需要抽取出尽可能多的关系。', history=[])
    return response.strip()

def ask_utils(text,model):
    text = text.strip()
    if model == 'GPT-3.5':
        return ask_GPT(text)
    elif model == 'chatglm-6b':
        return ask_glm(text)
    
gr.close_all()
demo = gr.Interface(fn=ask_utils,inputs=[gr.Textbox(lines=3,placeholder='请在此处输入文本'),gr.Radio(['GPT-3.5','chatglm-6b'])],
                    outputs= 'text',
                    examples= [['史蒂芬·斯皮尔伯格是美国著名电影导演，曾三次获得奥斯卡，因为执导了辛德勒的名单，拯救大兵瑞恩和斯坦利·库布里克的奇幻之旅','chatglm-6b'],['周华健演唱的刀剑如梦是一首非常好听的歌','chatglm-6b']],
                    title='基于GPT-3.5关系抽取',
                    description='在"text"框输入待分析段落,自动抽取文中关系')
demo.launch(server_name='0.0.0.0')
