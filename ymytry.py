import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import gensim.downloader as api
import time
import sys
import os
import json
bert_version = '/root/Desktop/Cloze-Test_Pytorch_BERT-master/bert-based-uncased'
tokenizer = BertTokenizer.from_pretrained(os.path.join(bert_version,'vocab.txt'))

path='/root/Desktop/Cloze-Test_Pytorch_BERT-master/ELE/dev'
f = json.load(open(os.path.join(path,'dev0001.json'), "r",encoding='UTF-8'))
texts=f['article']
choices=f['options']
answers=f['answers']
tokenizer = BertTokenizer.from_pretrained(os.path.join(bert_version,'vocab.txt'))
tokenized_text = tokenizer.tokenize(texts)
print(tokenized_text)
mask_positions = []
for i in range(len(tokenized_text)):#查找出是多少位单词需要填充
    if tokenized_text[i] == '_':
        tokenized_text[i] = '[MASK]'
        #print(i)
        #print(mask_positions)
        mask_positions.append(i)
for n in range(20):   #将answer中ABCD转换为0123
    answers[n]=ord(answers[n])-65
    #print(answers)
print('answers',answers)
print('mask_positions',mask_positions)
predicted_token = ''
acc = 0#正确数量
n=0#第n个answer-1
accuracy =0 #正确率
token_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor([token_ids])
#for mask_pos,n in zip(mask_positions,range(20)):
for mask_pos in mask_positions:
    # Convert tokens to vocab indices
    
    candidates_ids = tokenizer.convert_tokens_to_ids(choices[n])
    print(candidates_ids)
    model = BertForMaskedLM.from_pretrained(bert_version)
    model.eval()
    #candidates=choices[n]
    #candidates_ids = tokenizer.convert_tokens_to_ids(candidates)
    # print('tokens_tensor: ''\n',tokens_tensor)
    # Call BERT to predict token at this position
    #try:
     # predictions = model(tokens_tensor)[0, mask_pos,candidates_ids]
    #except RuntimeError:
    #  sys.exit('Oops! Sorry for your input 1-20 articles are too long. Try to decrease your sentences.')
   # else:
    segments_ids = [0] * len(tokenized_text)
    #tokens_tensor = torch.tensor([token_ids])
    segments_tensors = torch.tensor([segments_ids])
    predictions = model(tokens_tensor, segments_tensors)
    predictions_candidates = predictions[0, mask_pos,candidates_ids]
    # print("type.predictions:",type(predictions))
    print("predictions:"'\n',predictions)
    print('predictions_candidates',predictions_candidates)
    predicted_index = torch.argmax(predictions_candidates).item()
    print('predicted_index''\n',predicted_index)
    #predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    predicted_token=choices[n][predicted_index]
    print('predicted_token:''\n',predicted_token)
    tokenized_text[mask_pos] = predicted_token
    if tokenized_text[mask_pos]==choices[n][answers[n]]:
        print('yes')
        acc=acc+1
    accuracy=acc/20  #正确率
    n=n+1
for mask_pos in mask_positions:
    tokenized_text[mask_pos] = "_" + tokenized_text[mask_pos] + "_"#将原文预测答案填上
print(tokenized_text)#输出token后的原文
temp=tokenizer.convert_tokens_to_string(tokenized_text)#将token转换为原文
print(temp)
print('accuracy is %f' %(accuracy))
