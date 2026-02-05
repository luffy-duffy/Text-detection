from tqdm import tqdm
import re
from utils import *
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import numpy as np
from imblearn.over_sampling import RandomOverSampler

from lightgbm import LGBMClassifier

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

from snownlp import SnowNLP

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# BERT 数据集定义
class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'label': torch.tensor(int(self.labels[idx]), dtype=torch.long)
        }


# BERT + MLP 分类器
class BERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled = outputs.pooler_output
        return self.classifier(self.dropout(pooled))


def safe_get_sentiment(text):
    if not text.strip():
        return 0.5  # 中性值
    try:
        return SnowNLP(text).sentiments
    except ZeroDivisionError:
        return 0.5  # 处理全是停用词的文本


BULLY_KEYWORDS = [
    "废物", "滚", "蠢", "傻", "弱智", "脑残", "贱", "畜生", "神经病", "diao", "屌",
    "渣", "吐", "破", "nmsl", "NMSL", "杀", "cnm", "CNM", "艹", "日", "王八", "tm", "TM", "尼玛",
    "垃圾", "sb", "SB", "辣鸡", "稀烂", "败笔", "拉低水平", "拖后腿", "恶心", "不要个碧莲", "最差",
    "丁8", "丁丁", "JB", "jb", "没B数", "没b数", "婊", "蛆", "栽种", "Bitch", "bitch",
    "杂种", "狗杂种", "贱种", "死妈", "司马", "biss", "孤儿", "脑瘫", "wdnmd", "WDNMD",
    "烂的一逼", "药店碧莲", "唾弃", "狠毒", "忘恩负义", "差劲", "糊穿地心", "肛门", "脸皮厚", "呕",
    "操你妈", "草泥马","傻逼", "煞笔"
]

GROUP_SENSITIVE_KEYWORDS = [
    "狗", "死胖子", "死gay", "死黑鬼", "娘炮", "你妈", "马", "驴",
    "穷鬼", "窝囊废", "脑瘫", "残疾", "鸡", "女的都这样", "丑八怪", "日本鬼子", "棒子", "黑鬼",
    "女拳", "蝈蝻", "小日本", "台蛙", "港灿", "乡下人"
    "母狗",  "黑鬼", "白皮猪", "阿三", "台蛙", "港独"
    "捞佬", "乡下狗",  "倪哥", "白左", "田园女权","肥猪", "丑八怪", "矮冬瓜", "死胖子",
    "丑逼", "秃驴", "残疾", "婊子养的"
    "英年早逝", "你家人", "你亲妈", "你爹"
]

NEGATIVE_PHRASES = [
    "看不下去", "尬", "烦死了", "下线", "作", "厌"
    "硬凑", "败笔", "丑", "傻", "寡妇", "逼", "受不了", "不行", "快进", "别活着了", "什么玩意"
    "嗑药", "不是人" "人肉", "冲了他", "安排他", "搞他", "扒皮",
    "挂人", "一起攻击", "举报到封号", "求扩散", "搞臭", "大家一起骂", "网暴"
    "打死", "杀死", "砍死", "弄死", "炸死", "死全家", "不得好死", "千刀万剐", "下地狱",
    "索命", "尸体", "厉鬼索命", "病魔战胜你"
]

INTENSIFIERS = ["???", "......", "！！", "。。。", "？？？？", "！！！！", "？？？！", "！！！？？"]


# 特征提取函数
def extract_additional_features(texts, fit=False, vectorizer=None, tfidf_max_features=300):
    url_pattern = re.compile(r'https?://|www\.|\.com|\.cn|\.net')

    all_feature_rows = []

    for text in texts:
        # URL数量
        url_count = len(url_pattern.findall(text))

        # 情感倾向
        try:
            sentiment = SnowNLP(text).sentiments
        except ZeroDivisionError:
            sentiment = 0.5

        # 词典匹配
        bully_count = sum(text.count(word) for word in BULLY_KEYWORDS)
        group_count = sum(text.count(word) for word in GROUP_SENSITIVE_KEYWORDS)
        neg_phrase_count = sum(text.count(word) for word in NEGATIVE_PHRASES)
        intensifier_count = sum(text.count(word) for word in INTENSIFIERS)

        all_feature_rows.append([
            url_count,
            sentiment,
            bully_count,
            group_count,
            neg_phrase_count,
            intensifier_count,
        ])

    manual_features = np.array(all_feature_rows)  # N×7

    # TF-IDF 特征拼接
    if vectorizer is None or fit:
        vectorizer = TfidfVectorizer(max_features=tfidf_max_features)
        tfidf_matrix = vectorizer.fit_transform(texts).toarray()
    else:
        tfidf_matrix = vectorizer.transform(texts).toarray()

    # 合并
    final_features = np.hstack([manual_features, tfidf_matrix])

    return final_features, vectorizer


# 读取数据集
def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text_data = f.readlines()

    dataset = [s.strip().split('\t', 1) for s in text_data]
    dataset = [data for data in dataset if len(data) == 2 and data[1].strip()]

    tag = [data[0] for data in dataset]
    text = [data[1] for data in dataset]

    return tag, text


# 将数据集划分为训练集和测试集
def divide_dataset(tag, vector, text):
    train_vector, test_vector, train_tag, test_tag, train_text, test_text = train_test_split(vector, tag, text, test_size=0.5, random_state=42)

    return train_vector, test_vector, train_tag, test_tag,train_text,test_text


# 文本清洗
def clean_text(dataset):
    cleaned_text = []
    for text in tqdm(dataset, desc='Cleaning text'):
        clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        cleaned_text.append(clean.strip())
    return cleaned_text


# 停用词处理和文本分割
def tokenize_and_remove_stopwords(dataset):
    stopwords_file = 'D:/本科全部东西/大数据原理与技术/大作业/代码/dataset/dataset_my.txt'
    with open(stopwords_file, 'r', encoding='utf-8') as file:
        stopwords = {line.strip() for line in file}

    tokenized_text = []
    for text in tqdm(dataset, desc='Tokenizing and removing stopwords'):
        cleaned_text = ''.join([char for char in text if char not in stopwords and re.search("[\u4e00-\u9fa5]", char)])
        tokenized_text.append(cleaned_text)

    return tokenized_text


# 为每个汉字生成初始特征向量
def generate_w2v_vectors(tokenized_text, d=100):
    model = Word2Vec(sentences=tokenized_text, vector_size=d, window=5, min_count=1, sg=0)
    word_vectors = model.wv

    w2v_vectors = {}
    for tokens in tqdm(tokenized_text, desc='Generating word vectors'):
        for word in tokens:
            if word not in w2v_vectors.keys():
                w2v_vectors[word] = word_vectors[word]

    return w2v_vectors


# 为语料库中不曾存在的汉字生成字符向量并动态更新语料库
def update(w2v_vectors, text, character, d=100):
    model = Word2Vec(sentences=text, vector_size=d, window=5, min_count=1, sg=0)
    word_vectors = model.wv
    w2v_vectors[character] = word_vectors[character]

    return w2v_vectors


# 根据字符相似性网络生成最终的字嵌入向量
def generate_char_vectors(chinese_characters, w2v_vectors, sim_mat, text, chinese_characters_count, threshold=0.6):
    char_vectors = {}
    for i in tqdm(range(len(chinese_characters)), desc='Generating char vectors'):
        character = chinese_characters[i]
        similar_group = []
        for j in range(len(sim_mat[i])):
            if sim_mat[i][j] >= threshold:
                similar_group.append(chinese_characters[j])
        sum_count = 0
        emb = np.zeros_like(w2v_vectors[list(w2v_vectors.keys())[0]])  # 初始化一个全零向量
        for c in similar_group:
            if c not in w2v_vectors.keys():
                update(w2v_vectors, text, c)
            emb += chinese_characters_count[c] * w2v_vectors[c]
            sum_count += chinese_characters_count[c]
        emb /= sum_count if sum_count else 1  # 避免除以0
        char_vectors[character] = emb

    return char_vectors


# 根据字嵌入向量生成句子嵌入向量
def generate_sentence_vectors(texts, char_vectors, d=100):
    sentence_vectors = []
    for text in tqdm(texts, desc='Generating sentence vectors'):
        alpha = np.zeros((len(text), len(text)))
        for i in range(len(text)):
            for j in range(len(text)):
                alpha[i][j] = alpha[j][i] = np.dot(char_vectors[text[i]], char_vectors[text[j]]) / np.sqrt(d)

        alpha_hat = np.zeros_like(alpha)
        for i in range(len(text)):
            for j in range(len(text)):
                alpha_hat[i][j] = alpha_hat[j][i] = np.exp(alpha[i][j]) / np.sum(alpha[i])

        m = np.zeros((d,))  # 初始化一个全零向量
        for i in range(len(text)):
            mi = np.zeros((d,))
            for j in range(len(text)):
                mi += alpha_hat[i][j] * char_vectors[text[j]]
            m += mi
        sentence_vectors.append(m / d)

    return sentence_vectors


# 文本分类
def spam_classification(train_tags, train_word_vectors, test_word_vectors):
    # oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    # train_word_vectors, train_tags = oversampler.fit_resample(train_word_vectors, train_tags)

    # clf = LogisticRegression(max_iter=1000)

    # clf = XGBClassifier(
    #     n_estimators=400,
    #     max_depth=6,
    #     learning_rate=0.08,
    #     subsample=0.9,
    #     colsample_bytree=0.9,
    #     scale_pos_weight=2.84,
    #     reg_alpha=0.2,
    #     reg_lambda=2.0,
    #     use_label_encoder=False,
    #     eval_metric='logloss',
    #     random_state=42
    # )

    # clf = SGDClassifier(
    #     loss='hinge',  # SVM
    #     class_weight='balanced',
    #     max_iter=1000,
    #     tol=1e-3
    # )

    # clf = RandomForestClassifier(
    #     n_estimators=300,
    #     max_depth=10,
    #     class_weight='balanced_subsample',
    #     random_state=42
    # )

    # clf = LGBMClassifier(
    #     num_leaves=64,
    #     n_estimators=300,
    #     learning_rate=0.1,
    #     class_weight='balanced'
    # )

    # 调参
    clf = LGBMClassifier(
        num_leaves=31,
        n_estimators=100,
        learning_rate=0.05,
        class_weight='balanced',
        max_depth=7,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        objective='binary',
        metric='binary_logloss'
    )

    clf.fit(np.array(train_word_vectors), np.array(train_tags))

    # predictions = clf.predict(test_word_vectors)

    predictions = clf.predict_proba(test_word_vectors)[:, 1]

    return predictions


def evaluation(test_tags, predictions):
    # 输出混淆矩阵和分类报告
    cm = confusion_matrix(np.array(test_tags), np.array(predictions))
    print("混淆矩阵:")
    print(cm)

    report = classification_report(np.array(test_tags), np.array(predictions))
    print("分类报告:")
    print(report)


if __name__ == "__main__":
    tag, text = read_data('D:/本科全部东西/大数据原理与技术/大作业/代码/dataset/dataset_my.txt')

    chinese_characters, chinese_characters_count, chinese_characters_code = count_chinese_characters(text, 'D:\本科全部东西\大数据原理与技术\大作业\代码\dataset\hanzi.txt')
    sim_mat = compute_sim_mat(chinese_characters, chinese_characters_code)

    cleaned_text = clean_text(text)
    tokenized_text = tokenize_and_remove_stopwords(cleaned_text)
    w2v_vectors = generate_w2v_vectors(tokenized_text)
    char_vectors = generate_char_vectors(chinese_characters, w2v_vectors, sim_mat, text, chinese_characters_count)
    sentence_vectors = generate_sentence_vectors(tokenized_text, char_vectors)

    # 特征工程
    extra_features, tfidf_vectorizer = extract_additional_features(cleaned_text, fit=True)
    full_vectors = np.hstack([sentence_vectors, extra_features])

    # train_vectors, test_vectors, train_tag, test_tag = divide_dataset(tag, sentence_vectors)
    train_vectors, test_vectors, train_tag, test_tag,train_text,test_text = divide_dataset(tag, full_vectors,text)

    train_tag = np.array(train_tag, dtype=int)
    test_tag = np.array(test_tag, dtype=int)

    # predictions = spam_classification(train_tag, train_vectors, test_vectors)
    #
    # evaluation(test_tag, predictions)

    # BERT 模型训练
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    bert_model = BERTClassifier().to("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = BERTDataset(train_text, train_tag, tokenizer)
    test_dataset = BERTDataset(test_text, test_tag, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    optimizer = torch.optim.AdamW(bert_model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        bert_model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to("cuda")
            attention_mask = batch['attention_mask'].to("cuda")
            labels = batch['label'].to("cuda")

            optimizer.zero_grad()
            outputs = bert_model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

    # BERT 模型预测
    bert_model.eval()
    proba_bert = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to("cuda")
            attention_mask = batch['attention_mask'].to("cuda")

            logits = bert_model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)[:, 1]
            proba_bert.extend(probs.cpu().numpy())

    proba_bert = np.array(proba_bert)

    # 融合预测
    proba_lgb = spam_classification(train_tag, train_vectors, test_vectors)
    final_proba = 0.6 * proba_lgb + 0.4 * proba_bert
    final_pred = (final_proba >= 0.5).astype(int)

    evaluation(test_tag, final_pred)