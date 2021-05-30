'''
Descripttion: 
Author: cjh (492795090@qq.com)
Date: 2021-05-29 14:06:15
LastEditTime: 2021-05-30 10:23:15
'''
import pickle

def generate_ngram(input_list, n):
    if n == 1:
        return [(token,) for token in input_list]
    
    for _ in range(n-1):
        input_list.insert(0, '<s>')
    
    input_list.append('</s>')
    sequences = [input_list[i:] for i in range(n)]
    
    return [ngram for ngram in zip(*sequences)]


def save_model(model, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(model, fw)


def load_model(filename):
    with open(filename, 'rb') as fr:
        model = pickle.load(fr)
    return model

if __name__ == '__main__':
    text = u'<BEG>' + '中文文本自动校对技术的研究与实现' + u'<END>'
    print(generate_ngram(list(text), 3))
    