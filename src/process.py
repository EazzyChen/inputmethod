# 一,数据预处理

import pandas as pd
import config
import jieba
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def process():
    print("开始处理数据...")
    # 1. 读取文件
    df = pd.read_json(config.RAW_DATA_DIR / "synthesized_.jsonl", 
                      lines=True, 
                      orient='records').sample(frac=0.1, random_state=42) # 只抽10%数据用
    # print(df['dialog'])

    # 2. 提取句子（用 split('：', 1) 只切第一处冒号，正文里再有冒号也不会截断）
    sentences = []
    for dialog in df['dialog']:
        for sentence in dialog:
            sentences.append(sentence.split('：', 1)[1] if '：' in sentence else sentence)

    print(f'句子总数: {len(sentences)}')

    # 3. 划分数据集
    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2)
 

    # 4. 构建词表: 词 -> 序号
    vocab = set()
    for sentence in tqdm(train_sentences, desc='构建词表'): # tqdm: 进度条封装
        vocab.update(jieba.lcut(sentence))
    vocab_list = ['<unk>'] + list(vocab)
    print(f'词表大小:{len(vocab_list)}')

    # 5.保存词表
    with open(config.MODELS_DIR / 'vocab.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab_list))

    # 6.构建训练集: 一小段句子(上文) -> 下一个词
    word2index = {word: index for index, word in enumerate(vocab_list)}
    indexed_train_sentences = [[word2index.get(token,0) for token in jieba.lcut(sentence)] for sentence in train_sentences]
    train_dataset = []
    for sentence in tqdm(indexed_train_sentences, desc='构建训练集'):
        for i in range(len(sentence) - config.SEQ_LEN):  # 句子长度 - 上文窗口大小
            input = sentence[i: i + config.SEQ_LEN]  # 已输入句子（窗口）预测下一个词
            target = sentence[i+config.SEQ_LEN]
            train_dataset.append({'input': input, 'target': target})

    print(train_dataset[0: 3])
    
    # 7. 保存训练集
    pd.DataFrame(train_dataset).to_json(config.PROCESSED_DATA_DIR/'train.jsonl', orient='records', lines=True)

    # 8. 测试集，逻辑和训练集一样
    indexed_test_sentences = [[word2index.get(token,0) for token in jieba.lcut(sentence)] for sentence in test_sentences]
    test_dataset = []
    for sentence in tqdm(indexed_test_sentences, desc='构建测试集'):
        for i in range(len(sentence) - config.SEQ_LEN):
            input = sentence[i: i + config.SEQ_LEN]
            target = sentence[i+config.SEQ_LEN]
            test_dataset.append({'input': input, 'target': target})

    pd.DataFrame(test_dataset).to_json(config.PROCESSED_DATA_DIR/'test.jsonl', orient='records', lines=True)


    print('数据处理完成...')

    

    



if __name__ == '__main__':
    print(__file__)
    process()

