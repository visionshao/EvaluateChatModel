import os
import datasets
import matplotlib.pyplot as plt
import matplotlib as mpl
import datasets
print(mpl.get_cachedir())
#
plt.rc("font",family="AR PL UKai CN") ###修改了这一行

import seaborn as sns


class Evaluator:

    def __init__(self, name) -> None:
        self.name = name

    def evaluate(self, model, data):
        raise NotImplementedError

class IntentAlignmentEvaluator(Evaluator):
    
    def __init__(self, name, keywords_file_path) -> None:
        super().__init__(name)
        self.keywords_file_path = keywords_file_path
        self.get_keywords()

    def get_keywords(self):
        keywords = []
        with open(self.keywords_file_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                keywords.append(line.split()[0])

        self.keywords = keywords

    def evaluate(self, model, tokenizer, input_text_list, device, gen_config, plot=False, score_type="attn"):
        total_topk_input_tokens_num = 0
        total_matched_keywords_num = 0
        for idx, input_text in enumerate(input_text_list):
            r = self.get_score(model, tokenizer, input_text, device, gen_config, score_type)
            if r is not None:
                print(r["input_text"] + r["gen_text"])
                print(r["topk_input_tokens"])
                matched_keywords = self.match_keywords_base(r["topk_input_tokens"])
                print()
                print(matched_keywords)
                print("*"*100)
                total_topk_input_tokens_num += len(r["topk_input_tokens"])
                total_matched_keywords_num += len(matched_keywords)
                if plot:
                    self.plot_heat_map(r["score_matrix"], r["gen_token_list"], r["last_user_token_list"], "heat_map_{}.png".format(idx))

        print("total_topk_input_tokens_num: {}".format(total_topk_input_tokens_num))
        print("total_matched_keywords_num: {}".format(total_matched_keywords_num))
        print("intent sensitivity: {}".format(total_matched_keywords_num/total_topk_input_tokens_num))
    
    def get_score(self, model, tokenizer, input_text, device, gen_config, score_type="attn"):
        r = None
        if score_type == "attn":
            r = self.get_attn_score(model, tokenizer, input_text, device, gen_config)
        return r

    def match_keywords_base(self, topk_input_tokens):

        match_keywords = []
        for token in topk_input_tokens:
            if token in self.keywords:
                match_keywords.append(token)
        return match_keywords
        
    def plot_heat_map(self, score_matrix, gen_token_list, last_user_token_list, save_path):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax = sns.heatmap(score_matrix, xticklabels=last_user_token_list, yticklabels=gen_token_list, cmap="YlGnBu")
        ax.set_xlabel("Input Tokens", fontsize=24)
        ax.set_ylabel("Generated Tokens", fontsize=24)
        plt.savefig(save_path)
        
    def get_attn_score(self, model, tokenizer, input_text, device, gen_config, topk=3):
        if gen_config is not None:
            # tokenize input_text and log its length
            inputs = tokenizer([input_text], return_tensors="pt", padding=True)
            input_token_list = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]) # list of input tokens
            last_user_sentence_len = len(tokenizer([input_text.split("\n")[-2]]).input_ids[0])-5 # length of last user sentence
            last_user_token_list = input_token_list[-last_user_sentence_len-4:-4]
            bs, context_tokens_num = inputs["input_ids"].shape

            # generate
            inputs.to(device)
            outputs = model.generate(**inputs, generation_config=gen_config, return_dict_in_generate=True, output_scores=True, output_attentions=True)
            # get generated sequences
            gen_sequences = outputs.sequences[0][context_tokens_num:]
            # print(gen_sequences)
            gen_text = tokenizer.decode(gen_sequences, skip_special_tokens=True)
            gen_token_list = tokenizer.convert_ids_to_tokens(gen_sequences)

            new_outputs = model(outputs.sequences, return_dict=True, output_attentions=True)

            attentions = new_outputs["attentions"]
            last_layer_attentions = attentions[-1] # last layer
            average_last_attention = last_layer_attentions.mean(0)

            # get key tokens in the last user sentence
            score_matrix = average_last_attention[context_tokens_num:, context_tokens_num-last_user_sentence_len-4:context_tokens_num-4].data.cpu().numpy()
            input_sum_scores = score_matrix.sum(0)
            # select topk input tokens
            topk_idx = input_sum_scores.argsort()[-topk:][::-1]
            # print topk input tokens
            topk_input_tokens = [last_user_token_list[ind] for ind in topk_idx]

            # set results dict
            results = {
                "input_text": input_text,
                "gen_text": gen_text,
                "last_user_token_list": last_user_token_list,
                "gen_token_list": gen_token_list,
                "score_matrix": score_matrix,
                "topk_input_tokens": topk_input_tokens
            }

            return results
