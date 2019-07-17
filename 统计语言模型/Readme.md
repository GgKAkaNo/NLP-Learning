## 三、NLP的语言模型（统计语言模型）
**统计语言模型：** 统计语言模型把语言（词的序列）看作一个随机事件，
并赋予相应的概率来描述其属于某种语言集合的可能性。给定一个词汇集合 V，对于一个由 V 中的词构成的序列S = ⟨w1, · · · , wT ⟩ ∈ Vn，统计语言模型赋予这个序列一个概率P(S)，来衡量S 符合自然语言的语法和语义规则的置信度。
简单的说，语言模型就是用来计算一个句子的概率的概率模型

![图片1](https://github.com/GgKAkaNo/NLP_tutorial/blob/master/统计语言模型/png/图片1.png)


### N-gram

N-Gram是基于一个假设:第n个词出现不前n-1个词相关,而不其他任何词不相关。(这也是隐马尔可夫当中的假设。)整个句子出现的概率就等于各个词出现的概率乘积。各个词的概率可以通过语料中统计计算得到。假设句子T是有词序列w1,w2,w3...wn组成,用公式表示N-Gram语言模型如下:
![图片2](https://github.com/GgKAkaNo/NLP_tutorial/blob/master/统计语言模型/png/图片2.png)
![图片3](https://github.com/GgKAkaNo/NLP_tutorial/blob/master/统计语言模型/png/图片3.png)
