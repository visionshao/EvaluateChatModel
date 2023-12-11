# EvaluateChatModel

- Aims
    - Evaluate the intent alignments ability of LLM Agent

- Ways
    - Get a score function between generated tokens and input tokens (last user sentence)
    - Choose the tokens with the topK sum of relevance scores in the last user sentence
    - Match these tokens with the keywords database and use the rate of matching to represent the intent recoginitiion rate

- Some key tips:
    - Chatglm's structure should be revised to output attention scores. The original version of chatglm won't output attention scores due to the computational efficiency.
    - Different generation configs may lead to different results, and it is better to set a good generation config first.
    - We select the last layer's attention and average each head attetion matrix to produce the final attention matrix.

- How to use:
    - Provide a txt contains words related to intents.
    - Input Construction: connect utterances with "\n" and append "\n" + "客服人员" at the end.
    - Please replace the original modeling_chatglm.py with the same file in our 'model' directory. Here we change it to also produce attention scores when using Pytorch version 2.0+.
    - More utilization details could be see in example.py

- To be done
    - Find a better attention layer (here is last layer).
    - Add other scores for chatglm like gradient (need float16), pd and so on.
    - Optimize the selection method of focused tokens in user sentence (here is topK)
    - Optimize the obtain

    



