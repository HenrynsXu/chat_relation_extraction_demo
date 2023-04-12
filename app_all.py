import gradio as gr
import openai

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("./chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("./chatglm-6b", trust_remote_code=True).quantize(4).half().cuda()
model = model.eval()

def ask_GPT(text):
    
    openai.api_key = 'API_KEY'
    prompt = f'对下面这段文本进行关系抽取：“{text}”。要求为提取其中的实体名称，并确定它们之间的关系。注意你只需要回答关系，回答的模式为“实体1-关系-实体2”，在一行内输出一条关系，并且不用注明“实体1”、“关系”、“实体2”等，你需要抽取出尽可能多的关系，并在回答时尽量采用主动语态。'
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = [{"role":"user","content":prompt}],
            temperature=0.5)
    ans=response['choices'][0]['message']['content']
    return ans.strip()

def ask_GPT_rel(text,rel):
    openai.api_key = 'API_KEY'
    prompt = f'对下面这段文本进行关系抽取：“{text}”。抽取出其中的{rel}关系。注意你只需要用一句通顺的话回答关系即可，不用回答实体是什么。回答不需要前缀“关系”等，你需要抽取出尽可能多的关系。'
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = [{"role":"user","content":prompt}],
            temperature=0.5)
    ans=response['choices'][0]['message']['content']
    return ans.strip()

def ask_glm_no_rel(text):
    response, _ = model.chat(tokenizer, f'对下面这段文本进行关系抽取：“{text}”。要求为提取其中的实体名称，并确定它们之间的关系。注意你只需要回答关系，回答的模式为“实体1-关系-实体2”，在一行内输出一条关系，并且不用注明“实体1”、“关系”、“实体2”等，你需要抽取出尽可能多的关系。', history=[])
    return response.strip()

def ask_glm_rel(text,rel):
    # prompt = f'对下面这段文本进行关系抽取：“{text}”。要求为提取其中的实体名称，并确定它们之间的关系。注意你只需要回答关系，回答的模式为“实体1-关系-实体2”，在一行内输出一条关系，并且不用注明“实体1”、“关系”、“实体2”等，你需要抽取出尽可能多的关系。'

    prompt = f'对下面这段文本进行关系抽取：“{text}”。抽取出其中的{rel}关系。注意你只需要用一句通顺的话回答关系即可，不用回答实体是什么。回答不需要前缀“关系”等，你需要抽取出尽可能多的关系。'
    response, _ = model.chat(tokenizer, prompt, history=[])
    
    
    return response.strip()

def ask_utils(text,model,rel):
    text = text.strip()
    if model == 'GPT-3.5':
        if not rel:return ask_GPT(text)
        else: return ask_GPT_rel(text,rel)
    elif model == 'chatglm-6b':
        if not rel:
            return ask_glm_no_rel(text)
        else:
            return ask_glm_rel(text,rel)
    
gr.close_all()
demo = gr.Interface(fn=ask_utils,inputs=[gr.Textbox(lines=3,placeholder='请在此处输入文本'),gr.Radio(['GPT-3.5','chatglm-6b']),gr.Textbox(placeholder='对glm模型请在此处输入待抽取的关系')],
                    outputs= 'text',
                    examples= [['史蒂芬·斯皮尔伯格是美国著名电影导演，曾三次获得奥斯卡，因为执导了辛德勒的名单，拯救大兵瑞恩和斯坦利·库布里克的奇幻之旅','chatglm-6b','导演-作品'],
                               ['周华健演唱的刀剑如梦是一首非常好听的歌','chatglm-6b','歌曲-评价'],
                               ['汉尼拔通常指两个实体。一是古代迦太基名将汉尼拔·巴卡，被誉为西方的四大军事家之一；二是托马斯·哈里斯所著小说《沉默的羔羊》的主人公汉尼拔·莱克特，他智商极高，却又令人毛骨悚然。','GPT-3.5','']],
                    title='基于LLM关系抽取',
                    description='在"text"框输入待分析段落,自动抽取文中关系')
demo.launch(server_name='0.0.0.0')
