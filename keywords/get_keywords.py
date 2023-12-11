import os
import json
import datasets
import jieba.analyse
from collections import Counter

data_dir = r'/mnt/sgnfsdata/tolo-03-97/weishao/Eval4LLM/data/JSON'

f = open("keywords.txt", "w")
words_list = []
for fname in os.listdir(data_dir):
    if fname.endswith(".json"):
        data_path = os.path.join(data_dir, fname)
        try:
            data = datasets.load_dataset("json", data_files=data_path)
            for item in data["train"]["会话"][0]:
                if item["场景"] is not None:
                    context = item["会话内容"]
                    role = item["角色"]
                    results = jieba.analyse.textrank(context, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
                    words_list += list(results)
        except Exception as e:
            print(e)
            print(data_path)
            continue

words_dict = dict(Counter(words_list))
for k, v in words_dict.items():
    f.write("{}\t{}\n".format(k, v))