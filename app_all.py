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
    prompt = f'对下面这段文本进行关系抽取：“{text}”。抽取出其中的{rel}关系。注意你只需要用一句通顺的话回答每一种关系即可，不用回答实体是什么。回答不需要前缀“关系”等，你需要抽取出尽可能多的关系，并在回答时尽量采用主动语态。。'
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

    prompt = f'对下面这段文本进行关系抽取：“{text}”。抽取出其中的{rel}关系。注意你只需要用一句通顺的话回答每一种关系即可，不用回答实体是什么。回答不需要前缀“关系”等，你需要抽取出尽可能多的关系。'
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
demo = gr.Interface(fn=ask_utils,inputs=[gr.Textbox(lines=3,placeholder='请在此处输入文本'),gr.Radio(['GPT-3.5','chatglm-6b']),gr.Textbox(placeholder='（可选）请在此处输入待抽取的关系')],
                    outputs= 'text',
                    examples= [['史蒂芬·斯皮尔伯格是美国著名电影导演，曾三次获得奥斯卡，因为执导了辛德勒的名单，拯救大兵瑞恩和斯坦利·库布里克的奇幻之旅','chatglm-6b','导演-作品'],
                               ['周华健演唱的刀剑如梦是一首非常好听的歌','chatglm-6b','歌曲-评价'],
                               ['汉尼拔通常指两个实体。一是古代迦太基名将汉尼拔·巴卡，被誉为西方的四大军事家之一；二是托马斯·哈里斯所著小说《沉默的羔羊》的主人公汉尼拔·莱克特，他智商极高，却又令人毛骨悚然。','GPT-3.5',''],
                               ['《毕业生》导演迈克·尼科尔斯执导，朱莉娅·罗伯茨、裘德·洛、娜塔莉·波特曼 、克莱夫·欧文主演，2004年美国上映。这部其他的不说，四位俊男美女当年的巅峰颜值，就值得一看。本部的情感线索挺复杂，朱莉娅·罗伯茨扮演的女摄影师坚强独立事业有成。遇到裘德·洛扮演的年轻作家，二人擦出了爱情的火花，女摄影师却选择嫁给了克莱夫·欧文扮演的医生。婚后仍然与年轻作家藕断丝连、情感暧昧。年轻作家与女摄影师分手之后，有个了新的女朋友，娜塔莉·波特曼扮演的年轻艳舞女郎。而更加让观众大感意外的是，女摄影师的丈夫医生和艳舞女郎还有着千丝万缕的瓜葛。','GPT-3.5',''],
                               ['刘伟强导演，全智贤、郑雨盛、李成宰主演，2006年上映。由韩国制片方投资，香港幕后团队与韩国明星联手打造的爱情动作片。冷血杀手和国际刑警，在抓捕与反抓捕的猫鼠游戏过程中，身不由己的爱上了同一位女画家，三人剪不断理还乱的情感纠葛，上演了一出忧伤的爱情大戏。身在他乡的女画家慧英，在陌生的都市中时常感到孤独和寂寞，每天收到陌生人送来的雏菊，成为了心生温暖的时刻。一路追踪杀手踪迹来到此处的刑警郑宇，因为无意间手中的一盆雏菊，被慧英误会成了别人。郑宇却在与慧英相遇的时候，不可救药的爱上了这个陌生女人。身负多宗命案的杀手朴义，特殊的身份，使他无法堂堂正正的追求自己心爱的女人，只能在暗处默默的守候着慧英。两男一女千头万绪的复杂情感，在命运的捉弄下走向难以预料的结局。','GPT-3.5','']],
                    title='基于LLM关系抽取',
                    description='在"text"框输入待分析段落（建议不宜过长）,抽取文中关系')
demo.launch(server_name='0.0.0.0',server_port=41321)
