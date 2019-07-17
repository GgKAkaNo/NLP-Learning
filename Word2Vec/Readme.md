
# Word2Vec
Word2Vec是Google在2013年开源的一款将词表征为实数值向量的高效工具，采用的模型有CBOW(Continuous Bag-Of-Words，即连续的词袋模型)和Skip-Gram 两种。
## CBOW

假定每个词都跟其相邻的词的关系最密切，换句话说每个词都是由相邻的词决定的（CBOW模型的动机）。CBOW模型的输入是某个词A周围的n个单词的词向量之和平均，输出是词A本身的词向量。
![1](https://github.com/GgKAkaNo/NLP_tutorial/blob/master/Word2Vec/png/1.jpg)

1.输入层：上下文单词的one-hot.{假设单词向量空间dim为V，上下文单词个数为C(滑动窗口)}

2.所有one-hot分别乘以共享的输入权重矩阵W. {V * N矩阵，N为自己设定的数，初始化权重矩阵W}

3.所得的向量 {因为是one-hot所以为向量} 相加求平均作为隐层向量, size为1 * N.

4.乘以输出权重矩阵W' {N * V}

5.得到向量 {1*V} 激活函数处理得到V-dim概率分布  {PS: 因为是one-hot嘛，其中的每一维都代表着一个单词}，概率最大的index所指示的单词为预测出的中间词（target word）

6.与true label的onehot做比较，误差越小越好。

其中，目标词出现的条件概率为：

![公式1](https://github.com/GgKAkaNo/NLP_tutorial/blob/master/Word2Vec/png/公式1.png)

> 其实这种计算方法就是使得越共现的词，向量乘积越大,乘积越大，则概率越大。最终会实现和某一个词的相关词和其都相似，就使得这些相关词向量更加相似。

定义loss function（一般为交叉熵代价函数），采用梯度下降算法更新W和W'。训练完毕后，输入层的每个单词与矩阵W相乘得到的向量的就是我们想要的词向量（word embedding），这个矩阵（所有单词的word embedding）也叫做look up table（其实这个look up table就是矩阵W自身），也就是说，任何一个单词的one-hot乘以这个矩阵都将得到自己的词向量。有了look up table就可以免去训练过程直接查表得到单词的词向量了。
![2](https://github.com/GgKAkaNo/NLP_tutorial/blob/master/Word2Vec/png/2.jpg)
![3](https://github.com/GgKAkaNo/NLP_tutorial/blob/master/Word2Vec/png/3.jpg)
![4](https://github.com/GgKAkaNo/NLP_tutorial/blob/master/Word2Vec/png/4.jpg)
![5](https://github.com/GgKAkaNo/NLP_tutorial/blob/master/Word2Vec/png/5.jpg)
![6](https://github.com/GgKAkaNo/NLP_tutorial/blob/master/Word2Vec/png/6.jpg)



## 多层softmax
普通的基于神经网络的语言模型输出层一般就是利用 softmax 函数进行归一化计算，这种直接 softmax 的做法主要问题在于计算速度。尤其是我们采用了一个较大的词汇表的时候，对大的词汇表做求和运算，softmax 的分母运算会非常慢，直接影响到了模型性能。

![7](https://github.com/GgKAkaNo/NLP_tutorial/blob/master/Word2Vec/png/7.jpeg)
![公式2](https://github.com/GgKAkaNo/NLP_tutorial/blob/master/Word2Vec/png/公式2.jpeg)

所以上面给出的示例是基于 Huffman 树的 Hierarchical Softmax （分级 softmax）输出层。跟大多数神经网络一样，CBOW 的训练方法仍然是基于损失函数的梯度计算方法，其目标函数如下：

![公式3](https://github.com/GgKAkaNo/NLP_tutorial/blob/master/Word2Vec/png/公式3.jpeg)

## 负采样


## skip-gram









