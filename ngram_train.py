'''
Descripttion: 
Author: cjh (492795090@qq.com)
Date: 2021-05-29 13:10:42
LastEditTime: 2021-06-09 08:49:35
'''
import  os
pwd_path = os.path.abspath(os.path.dirname(__file__))
from utils import generate_ngram as word_ngrams, save_model, load_model
from collections import Counter
from math import log2
from abc import ABC, abstractmethod

class TokenNode:
    """Trie node."""
    def __init__(self, word, count=0):
        self.word = word
        self.children = {}
        self.count = count

    def num_children(self):
        return len(self.children)
    
    def add_child(self, child):
        self.children[child.word] = child

    def has_child(self, word):
        return word in self.children
    
    def get_child(self, word):
        if self.has_child(word):
            return self.children[word]


class NGramLM(ABC):
    """Represent LM as a trie."""
    def __init__(self, n):
        self.n = n
        self.model_path = os.path.join(pwd_path, './model/ngram.model')
        if os.path.exists(self.model_path):
            self.root = load_model(self.model_path)
        else:
            self.root = TokenNode("")

    def train(self, file_path):
        """Train LM with the given file."""
        print("Training {}-gram models ({})...".format(self.n, self.name()))
        with open(file_path, 'r', encoding='utf8') as f:
            sentences = f.readlines()
        ngrams = [gram for sentence in sentences for gram in word_ngrams(input_list=list(sentence.strip().replace(' ','')), n=self.n)]
        counter = Counter(ngrams)
        for gram in counter:
            count = counter[gram]
            self.add_gram(gram, count)
            # handle lower-n grams at the end of the sentence
            if gram[-1] == "</s>":
                for i in range(self.n-1):
                    sub_gram = gram[i+1:]
                    self.add_gram(sub_gram, count)
        save_model(self.root, self.model_path)

    def add_gram(self, gram, count):
        """Add a new n-gram to the trie."""
        if not gram:
            return
        
        node = self.root
        for word in gram:
            node.count += count
            if not node.has_child(word):
                new_node = TokenNode(word)
                node.add_child(new_node)
            node = node.get_child(word)
        node.count += count # last word

    def get_vocab(self):
        """Return the vocabulary."""
        return self.root.children

    def get_vocab_size(self):
        """Return the size of vocabulary."""
        return self.root.num_children()

    def get_count(self, gram):
        """Given a n-gram as list of words, return its absolute count."""
        node = self.root
        for word in gram:
            node = node.get_child(word)
            if node is None:
                return 0
        return node.count

    def get_num_children(self, history):
        """Given a history, returns it N_{+1}(history%)."""
        node = self.root
        for word in history:
            node = node.get_child(word)
            if node is None:
                return 0
        return node.num_children()

    def perplexity(self, sentence, params=None):
        """Given a corpus (i.e., a list of raw sentences), return its perplexity."""
        L = 0
        M = 0
        ngrams = word_ngrams(sentence, self.n)
        for gram in ngrams:
            gram = list(gram)
            P_gram = self.estimate_smoothed_prob(gram[:-1], gram[-1], params)
            L += log2(P_gram)
        M += len(ngrams)
        return 2 ** (-L/M)

    def score_sentence(self, sentence, params=None):
        """Given a sentence, return score."""
        ngrams = word_ngrams(sentence, self.n)
        S = 0
        for gram in ngrams:
            gram = list(gram)
            P_gram = self.estimate_smoothed_prob(gram[:-1], gram[-1], params)
            S += log2(P_gram)
        # return -S / len(ngrams)
        return S

    @abstractmethod
    def estimate_smoothed_prob(self, history, word, params=None):
        pass

    @abstractmethod
    def name(self):
        pass

    def test(self):
        """Test whether or not the probability mass sums up to one."""
        precision = 10**-8
        histories = [[],['the'], ['in'], ['blue'], ['ahihi'], ['that', 'is'], ['that', 'ahihi'], ['ahihi', 'the'], ['ahihi', 'dongok']]
        for h in histories:
            if len(h) >= self.n:
                continue
            P_sum = sum(self.estimate_smoothed_prob(h, w) for w in self.get_vocab())
            print(str(h), P_sum)
            assert abs(1.0 - P_sum) < precision, 'Probability mass does not sum up to one for history ' + str(h)
        print('TEST SUCCESSFUL!')


class LidstoneLM(NGramLM):
    """N-gram LM with Lidstone smoothing."""
    def estimate_smoothed_prob(self, history, word, params=None):
        """Given a list of history words and the a word, return the lidstone prob P(word|history)."""
        alpha = params['alpha'] if params is not None and 'alpha' in params else 0.5
        return (alpha + self.get_count(history + [word])) / (alpha * self.get_vocab_size() + self.get_count(history))
    
    def name(self):
        return "Lidstone smoothing"


class AbsDiscountLM(NGramLM):
    """N-gram LM with absolute discounting smoothing."""
    def estimate_smoothed_prob(self, history, word, params=None):
        """Given a list of history words and the a word, return the absolute discounting prob P(word|history)."""
        d = params['d'] if params is not None and 'd' in params else 0.5
        
        P_lower = self.estimate_smoothed_prob(history[1:], word, params) if len(history) > 0 else 1 / self.get_vocab_size()

        N_history = self.get_count(history)
        if N_history == 0:
            return P_lower
        
        N_sequence = self.get_count(history + [word])
        lambda_factor = self.discounting_factor_lambda(history, d)
        return max(N_sequence - d, 0) / N_history + lambda_factor * P_lower
    
    def discounting_factor_lambda(self, history, d):
        """Compute the lambda factor in the absolute discounting smoothing method."""
        return d * self.get_num_children(history) / self.get_count(history)

    def name(self):
        return "absolute discounting smoothing"


def main():
    file_path = os.path.join(pwd_path,'./data/paper.txt')
    lm = AbsDiscountLM(n=3)
    if os.path.exists(lm.model_path):
        lm.root = load_model(lm.model_path)
    else:
        lm.train(file_path)
    test1 = '双粒度表示在字粒度基础上融合了词粒度模型校对'
    test2 = '双粒度表示在字粒度基础上融合了此粒度模型校对'
    print(lm.score_sentence(list(test1)))
    print(lm.score_sentence(list(test2)))
    print(lm.perplexity(list(test1)))
    print(lm.perplexity(list(test2)))
    return lm


if __name__ == "__main__":
    main()
