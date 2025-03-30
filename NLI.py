import pandas as pd
from time import time
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import nltk
from nltk.tokenize import word_tokenize
import string
import re
import csv

nltk.download('punkt')
nltk.download('punkt_tab')

# 读取数据集
def read_tsv(file_path):
    return pd.read_csv(file_path, sep='\t', header=None, names=['index', 'question', 'sentence', 'label'], quoting=csv.QUOTE_NONE)

train_df = read_tsv('QNLI/train_40.tsv')
dev_df = read_tsv('QNLI/dev_40.tsv')

# 数据预处理函数
def preprocess_text(text):
    text = text.lower()  # 转为小写
    text = re.sub(f"[{string.punctuation}]", "", text)  # 移除标点符号
    text = word_tokenize(text)  # 分词
    return text

# 应用预处理函数
train_df['question'] = train_df['question'].apply(preprocess_text)
train_df['sentence'] = train_df['sentence'].apply(preprocess_text)
dev_df['question'] = dev_df['question'].apply(preprocess_text)
dev_df['sentence'] = dev_df['sentence'].apply(preprocess_text)

# 构建词汇表
vocab = set()
for sentence in train_df['question'].tolist() + train_df['sentence'].tolist() + dev_df['question'].tolist() + dev_df['sentence'].tolist():
    vocab.update(sentence)
word2idx = {word: idx + 1 for idx, word in enumerate(vocab)}
word2idx['<PAD>'] = 0

# 将句子转换为索引序列
def convert_to_indices(sentence, word2idx):
    return [word2idx[word] if word in word2idx else 0 for word in sentence]

train_df['question_indices'] = train_df['question'].apply(lambda x: convert_to_indices(x, word2idx))
train_df['sentence_indices'] = train_df['sentence'].apply(lambda x: convert_to_indices(x, word2idx))
dev_df['question_indices'] = dev_df['question'].apply(lambda x: convert_to_indices(x, word2idx))
dev_df['sentence_indices'] = dev_df['sentence'].apply(lambda x: convert_to_indices(x, word2idx))
#print(train_df.iloc[1]['question'])
#print(train_df.iloc[1]['question_indices'])

# 将标签转换为数值
label2idx = {'entailment': 1, 'not_entailment': 0}
train_df['label'] = train_df['label'].map(label2idx)
dev_df['label'] = dev_df['label'].map(label2idx)

train_df = train_df.dropna(subset=['label'])
dev_df = dev_df.dropna(subset=['label'])

train_df['label'] = train_df['label'].astype(int)
dev_df['label'] = dev_df['label'].astype(int)

# 自定义数据集类
class NLIDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        question = torch.tensor(self.df.iloc[idx]['question_indices'])
        sentence = torch.tensor(self.df.iloc[idx]['sentence_indices'])
        label = torch.tensor(self.df.iloc[idx]['label']).long()  
        return question, sentence, label

# 数据对齐（padding）
def collate_fn(batch):
    questions, sentences, labels = zip(*batch)
    questions_padded = pad_sequence(questions, batch_first=True, padding_value=0)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return questions_padded, sentences_padded, labels

# 创建数据加载器
train_dataset = NLIDataset(train_df)
dev_dataset = NLIDataset(dev_df)

train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32, collate_fn=collate_fn)

# 转换到GPU运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


class NLIModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(NLIModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, question, sentence):
        question_embed = self.embedding(question)
        sentence_embed = self.embedding(sentence)
        
        _, (question_hidden, _) = self.lstm(question_embed)
        _, (sentence_hidden, _) = self.lstm(sentence_embed)
        
        question_hidden = torch.cat((question_hidden[-2], question_hidden[-1]), dim=1)
        sentence_hidden = torch.cat((sentence_hidden[-2], sentence_hidden[-1]), dim=1)
        
        combined = question_hidden * sentence_hidden
        output = self.fc(combined)
        
        return output

# 超参数
vocab_size = len(word2idx)
embed_size = 100
hidden_size = 128
output_size = 2

# 初始化模型、损失函数和优化器
model = NLIModel(vocab_size, embed_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    t1 = time()
    model.train()
    for questions, sentences, labels in train_loader:
        questions, sentences, labels = questions.to(device), sentences.to(device), labels.to(device)
        outputs = model(questions, sentences)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for questions, sentences, labels in dev_loader:
            questions, sentences, labels = questions.to(device), sentences.to(device), labels.to(device)
            outputs = model(questions, sentences)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    t2 = time()
    print(f'Epoch {epoch + 1}, Accuracy: {accuracy * 100:.2f}%, Running time: {t2 - t1:.2f}seconds')
