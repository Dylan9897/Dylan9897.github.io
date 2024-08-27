---
layout:     post   				    # 使用的布局（不需要改）
title:      Byte Pair Encoding 		# 标题 
subtitle:   BPE						#副标题
date:       2024-08-27 				# 时间
author:     BY handx				# 作者
header-img: img/post-bg-2015.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - transformer
    - LLM
---

# Byte-Pair-Encoding

## 1、简介

​	在自然语言处理（NLP）领域，处理输入数据是构建高效模型的关键步骤之一。以典型任务为例，当面对句子输入如**$I\ went\ to\ New\ York\ last\ week$**时，传统做法往往遵循简单的空格分隔原则，将句子拆分为独立的词汇单元，形成如下序列：
$$
[I,went,to,New,York,last,week]
$$
​	尽管这种处理方式在多数场景下表现良好，其局限性也日益凸显，主要体现在以下几个方面：

- **OOV（Out-of-Vocabulary）问题**

  最直接且显著的缺陷在于，该方法难以有效应对词汇表中的未登录词（OOV）问题。随着语言使用的多样性和动态变化，新词、缩写、专有名词等不断涌现，而这些新出现的词汇往往未包含在预定义的词汇表中。因此，简单依赖空格分隔并查找词汇表的方式，会导致大量有意义的词汇被忽略或错误处理，从而影响模型的准确性和泛化能力。

- **形态学信息缺失**

  另一个重要局限在于，该方法忽略了词汇间的形态学关系，如英文中的比较级和最高级形态变化。以$old$、$older$和$oldest$为例，这些词汇之间蕴含着丰富的语义和形态变化信息，但传统方法仅将它们视为独立的词汇单元，无法捕捉到它们之间的内在联系和变化规律。这种信息的缺失，限制了模型在理解和生成自然语言时的深度和广度。

​	在NLP任务中，面对传统分词方法所带来的OOV问题和形态学信息缺失的挑战，一种更为灵活和有效的解决方案被提出，即字节对编码（Byte Pair Encoding，简称BPE）算法。

## 2、BPE算法

​	BPE算法，作为一种高效的统计型无监督分词技术，其起源虽在于文本压缩领域，但现已深刻融入自然语言处理的核心，特别是作为现代预训练语言模型（BERT、GPT等）不可或缺的tokenizer编码手段。该算法的核心机制在于迭代式地合并语料库内最高频的字符对（或子词对），精心构建出一个紧凑且优化的子词词汇表，这一过程极大地增强了对输入文本的分词灵活性和适应性。

BPE算法的执行流程精炼地划分为两大阶段：

​	（1）**语料库学习与词库构建**：首先，以语料库作为输入，通过统计分析与合并操作，自动学习并构建出一个富含语义与形态学信息的子词词库。

​	（2）**tokenizer应用**：随后，基于构建的词库，对待处理的文本进行tokenizer（分词）操作。

算法流程：

- 给定子词词库
  $$
  \{A,B,C,...,a,b,c,...\}
  $$

- Repeat

  - 识别并选取训练语料库中最为频繁共现的两个相邻符号（或字符对/子词对），这两个符号在文本序列中紧密相连，共同出现的频率最高，例如$(A,B)$
  - 将频率最高的字符串$(A,B)$加入到词库中。
  - 将语料中所有的$(A,B)$替换成$AB$。

- 持续进行迭代，直至达到预设的迭代次数阈值，或者词汇表的大小满足既定的标准为止

## 3、代码实现

（1）下载数据集

```bash
wget http://www.gutenberg.org/cache/epub/16457/pg16457.txt
```

（2）代码实现

```python
import re
import collections

def get_vocab(filepath):
    """
    读取文件，并生成词表，每个单词按字符分隔，并加上结束符
    :param filepath:
    :return:
    """
    vocab = collections.defaultdict(int)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip("\ufeff").strip().split()
            for word in words:
                vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab

def measure_token_length(token):
    """
    获取token的长度
    :param token:
    :return:
    """
    if token[-4:] == '</w>':
        return len(token[:-4]) + 1
    else:
        return len(token)

class BpeObject():
    def __init__(self):
        pass

    def get_stats(self, vocab):
        """
        统计两两组合的字符 pairs 出现的频率
        :param vocab:
        :return:
        """
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    # def get_tokens(self, vocab):
    #     """
    #     统计 vocab 中每个字符的频率
    #     :param vocab:
    #     :return:
    #     """
    #     tokens = collections.defaultdict(int)
    #     for word, freq in vocab.items():
    #         word_tokens = word.split()
    #         for token in word_tokens:
    #             tokens[token] += freq
    #     return tokens

    def merge_vocab(self, pair, v_in):
        """
        将频率最高的字符串$(A,B)$加入到词库中，将语料中所有的$(A,B)$替换成$AB$。
        :param pair: 频率最高的字符串
        :param v_in: 词表
        :return:
        """
        v_out = {}
        bigram = re.escape(" ".join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def get_tokens_from_vocab(self, vocab, verbose=False):
        """
        统计 vocab 中每个字符的频率
        :param vocab:
        :return:
        """
        tokens_frequencies = collections.defaultdict(int)
        vocab_tokenization = {}
        for word, freq in vocab.items():
            if verbose:
                print(f"checkpoint for word is {word}")
                print(f"checkpoinf for freq is {freq}")
            word_tokens = word.split()
            for token in word_tokens:
                tokens_frequencies[token] += freq
            vocab_tokenization[''.join(word_tokens)] = word_tokens
        return tokens_frequencies, vocab_tokenization

    def tokenize(self, string, sorted_tokens, unknown_token="</u>"):
        """
        递归算法：对单词进行切分
        :param string:
        :param sorted_tokens:
        :param unknown_token:
        :return:
        """
        if string == "":
            return []
        if sorted_tokens == []:
            return unknown_token
        string_tokens = []
        for i in range(len(sorted_tokens)):
            token = sorted_tokens[i]
            token_reg = re.escape(token.replace('.', '[.]'))

            matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
            if len(matched_positions) == 0:
                continue

            substring_end_positions = [matched_position[0] for matched_position in matched_positions]

            substring_start_position = 0
            for substring_end_position in substring_end_positions:
                substring = string[substring_start_position:substring_end_position]
                string_tokens += self.tokenize(string=substring, sorted_tokens=sorted_tokens[i + 1:],
                                                    unknown_token=unknown_token)
                string_tokens += [token]
                substring_start_position = substring_end_position + len(token)
            remaining_substring = string[substring_start_position:]
            string_tokens += self.tokenize(string=remaining_substring, sorted_tokens=sorted_tokens[i + 1:],
                                                unknown_token=unknown_token)
            break
        return string_tokens


if __name__ == '__main__':
    from tqdm import tqdm
    # 读取文件
    vocab = get_vocab("data/pg16457.txt")
    # 初始化 bpe 函数
    bpe = BpeObject()

    tokens_frequencies, vocab_tokenization = bpe.get_tokens_from_vocab(vocab)

    print('All tokens: {}'.format(tokens_frequencies.keys()))
    print('Number of tokens: {}'.format(len(tokens_frequencies.keys())))
    print('==========')

    num_merges = 100
    for i in tqdm(range(num_merges)):
        pairs = bpe.get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = bpe.merge_vocab(best, vocab)
        tokens_frequencies, vocab_tokenization = bpe.get_tokens_from_vocab(vocab)

    test_word_list = ['mountains</w>','Ilikeeatingapples!</w>']

    # 对生成的词表进行排序
    sorted_tokens_tuple = sorted(tokens_frequencies.items(), key=lambda item: (measure_token_length(item[0]), item[1]),
                                 reverse=True)
    sorted_tokens = [token for (token, freq) in sorted_tokens_tuple]
    for elem in test_word_list:
        if elem in vocab_tokenization:
            print(f"after tokenizer, result is {vocab_tokenization[elem]}")
        else:
            word = bpe.tokenize(string=elem, sorted_tokens=sorted_tokens, unknown_token='</u>')
            print(f"after tokenizer, result is {word}")
```

