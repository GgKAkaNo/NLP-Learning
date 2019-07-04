# code by Tae Hwan Jung @graykode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

dtype = torch.FloatTensor

sentences = [ "i like dog", "i love coffee", "i hate milk"]
word_list = " ".join(sentences).split()#切片

word_list = list(set(word_list))

word_dict = {w: i for i, w in enumerate(word_list)}
print(word_dict)
number_dict = {i: w for i, w in enumerate(word_list)}

n_class = len(word_dict) # 词汇量
print(n_class)

# NNLM Parameter
n_step = 2 # n-1 
n_hidden = 2 # h 
m = 8 # m 

def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        
        target = word_dict[word[-1]]

        input_batch.append(input)
        target_batch.append(target)
    print('input_batch')
    print(input_batch)
    print(target_batch)
    return input_batch, target_batch

# Model
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(n_class, m)
        self.W = nn.Parameter(torch.randn(n_step * m, n_hidden).type(dtype))
        #self.W = nn.Parameter(torch.randn(n_step * m, n_class).type(dtype))
        self.p = nn.Parameter(torch.randn(n_hidden).type(dtype))
        self.U = nn.Parameter(torch.randn(n_hidden, n_class).type(dtype))
        self.q = nn.Parameter(torch.randn(n_class).type(dtype))

    def forward(self, X):
        X = self.C(X)
        X = X.view(-1, n_step * m) # [batch_size, n_step * n_class]
        tanh = torch.tanh(self.p + torch.mm(X, self.W)) # [batch_size, n_hidden]
        #output = self.q + torch.mm(X, self.W) + torch.mm(tanh, self.U) # [batch_size, n_class]
        output = self.q + torch.mm(tanh,self.U)
        return output

model = NNLM()

criterion = nn.CrossEntropyLoss()#交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

input_batch, target_batch = make_batch(sentences)
input_batch = Variable(torch.LongTensor(input_batch))
target_batch = Variable(torch.LongTensor(target_batch))

# Training
for epoch in range(5000):

    optimizer.zero_grad()
    output = model(input_batch)

    # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, target_batch)
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()

# Predict
predict = model(input_batch).data.max(1, keepdim=True)[1]

# Test
print(predict)
print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])


# for i, label in enumerate(word_list):
#     W, WT = model.parameters()
#     x,y = float(W[i][0]), float(W[i][1])
#     plt.scatter(x, y)
#     plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
# plt.show()
