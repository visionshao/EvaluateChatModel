import os
from evaluator import IntentAlignmentEvaluator
from models.chatglm.tokenization_chatglm import ChatGLMTokenizer
from models.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel, GenerationConfig

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datasets
print(mpl.get_cachedir())
#
plt.rc("font",family="AR PL UKai CN") ###修改了这一行

import seaborn as sns


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

device = "cuda:3"
data_path = r'/mnt/sgnfsdata/tolo-03-97/weishao/Eval4LLM/data/JSON/INFOOK0_20220101_091250060_0002.json'
model_path = r'/mnt/sgnfsdata/tolo-03-97/weishao/Eval4LLM/models/chatglm'

# initialize evaluator
MyEvaluator = IntentAlignmentEvaluator("IntentAlignmentEvaluator", r"/mnt/sgnfsdata/tolo-03-97/weishao/Eval4LLM/keywords/keywords.txt")

# initialize generation config
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
gen_config = GenerationConfig()
gen_config.max_length = 1024
gen_config.num_beams = 4
gen_config.early_stopping = True
gen_config.max_new_tokens = 128
gen_config.eos_token_id = 2
gen_config.pad_token_id = 0
gen_config.do_sample = True
gen_config.top_p = 0.8
gen_config.temperature = 0.8
gen_config.output_attentions = True
gen_config.return_dict_in_generate = True

# initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = ChatGLMForConditionalGeneration.from_pretrained(model_path, config=config, trust_remote_code=True)
model = model.half().to(device)

context_length = 2
data = datasets.load_dataset("json", data_files=data_path)
input_texts = ["{}:{}".format(item["角色"], item["会话内容"]) for item in data["train"]["会话"][0]]
# print(input_texts)
context_input_texts = [("\n".join(input_texts[max(0, i-context_length):i]) + "\n" + "客服人员:", input_texts[i]) for i, item in enumerate(input_texts) if i & 1 == 0 and i >= context_length]
context_input_texts = [item[0] for item in context_input_texts]
MyEvaluator.evaluate(model, tokenizer, context_input_texts, device, gen_config, plot=False, score_type="attn")