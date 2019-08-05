## 1.什么是Glove

GloVe的全称叫（Global Vector for Word Representation），它是一个基于全局词频统计（count-based & overall statistics）的词表征（word representation）工具。它可以把一个单词表达成一个由实数组成的向量，这些向量捕捉到了单词之间一些语义特性，比如相似性（similarity）、类比性（analogy）等。我们通过对向量的运算，可以计算出两个单 词之间的语义相似性。

做word embedding的方法，在2014年的时候主要有两种。一种是Matrix Factorization Methods（矩阵分解方法），一种是Shallow Window-Based Methods（基于浅窗口的方法）。关于Shallow Window-Based Methods，一个典型代表就是word2vec的方法，我在博客[5]已经解析过了。Matrix Factorization Methods这种方法，大致可以理解为是一种基于统计的方法，下面以简单例子说明。

一个简单的例子：Window based co-occurrence matrix（基于窗口的共现矩阵）。滑动窗口为1（通常使用5-10），矩阵具有对称性. 
样本数据集：
  1. I like deep learning.
  2. I like NLP.
  3. I enjoy flying.
  
构建的共现矩阵如下图所示：
  ![图像1](Glove/图像2)
  
但是如此构建共现矩阵，有以下问题
  1. 随着词汇增多，向量的大小会变得很大。
  2. 非常高维，需要大量存储空间。
  3. 随后的分类模型具有稀疏性的问题。（注：因为维数高，又稀疏，模型分类不易）
  4. 模型不够健壮（robust）。

我们可以用svd（singular value decomposition，奇异值分解），如图2：

![图像2](Glove/图像2)

奇异值类似主成分，在实际应用中，往往取top k个奇异值就能够表示绝大部分信息量，因此SVD经常拿来做损失较小的有损压缩：
SVD的几何意义实际上是通过线性变换来找到最能表达矩阵信息的一组正交基，原1*n维词向量在取得top k 奇异值后，可以用1*k维向量来表示该word，进而实现word embedding的效果。
从直观上我们发现，U矩阵和V矩阵可以近似来代表X矩阵，换据话说就是将A矩阵压缩成U矩阵和V矩阵，至于压缩比例得看当时对S矩阵取前k个数的k值是多少。

SVD（奇异值分解）的简单python例子如下图图3。其中数据集依然是：1. I like deep learning. 2. I like NLP. 3. I enjoy flying. 。然后在图4中，输出U矩阵的前两为最大的单个值。

![图像3](Glove/图像3)

![图像4](Glove/图像4)

将基于计数的和基于直接预测的两种word embedding方法对比：
基于计数的方法的优点是： 
  1. 训练非常迅速。 
  2. 能够有效的利用统计信息。 

缺点是：
  1. 主要用于获取词汇之间的相似性（其他任务表现差） 
  2. 给定大量数据集，重要性与权重不成比例。

基于预测的方法的优点是：
  1. 能够对其他任务有普遍的提高。 
  2. 能够捕捉到含词汇相似性外的复杂模式。 

缺点是：
  1. 需要大量的数据集。 
  2. 不能够充分利用统计信息。

以上两种方法都有优缺点，GloVe就是将两种方法的优点结合起来。

## 2.Glove如何实现

  1. 根据语料库（corpus）构建一个共现矩阵𝑿，设词表大小为V，共现矩阵将是一个方阵，第i行j列表示以第i个中心词w_𝒊,第j个背景词w_𝒋出现的次数。
假设有上下文：
An apple a day keeps an apple a day
设定滑动窗口为2，则中心词-背景词对有：

![图像5](Glove/图像5)

![图像6](Glove/图像6)

2.更新共现矩阵有：
定义共现矩阵的第i行的和为：

𝑿_i= ∑_(𝒋=𝟏)^𝑽▒𝑿_(𝒊,𝒋) 

之后我们有条件概率，对第j列对应的词出现在第i行上下文中的条件概率为：

  𝑷_(𝒊,𝒋)=𝑷(𝒋|𝒊)=  𝑿_(𝒊,𝒋)/𝑿_𝒊 
对于某个词w_𝒌,他在第i行或者第j行上下文出现的条件概率的比值为：
𝑷_(𝒊,𝒌)/𝑷_(𝒋,𝒌) 

## 3.Glove如何训练
## 4.Glove与Word2Vec的区别和联系
