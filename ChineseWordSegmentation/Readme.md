## 分词
==中文分词==（Chinese Word Segmentation）指将一个汉语序列切分成一个一个单独的词。分词就是将连续的字序列按照一定的规范重新组合成词序列的过程。

## 词性标注
==词性标注==(Part-of-Speech tagging 或POS tagging),又称词类标注或者简称标注,是指为分词结果中的每个单词标注一个正确的词性的程
序,也即确定每个词是名词、动词、形容词或其他词性的过程。在汉语中,词性标注比较简单,因为汉语词汇词性多变的情况比较少见,大多词语只有一个词性,或者出现频次最高的词性远远高于第二位的词性。据说,只需选取最高频词性,即可实现80%准确率的中文词性标注程序。

不同的工具词性标注不一定一样。

## 命名实体识别
==命名实体识别==(Named Entity Recognition,简称NER),又称作“专名识别”,是指识别文本中具有特定意义的实体,主要包括人名、地名、机构名、专有名词等。一般来说,命名实体识别的任务就是识别出待处理文本中三大类(实体类、时间类和数字类)、七小类(人名、机构名、地名、时间、日期、货币和百分比)命名实体。
在不同的顷目中,命名实体类别具有不同的定义。

### NLTK 

Natural Language Toolkit，自然语言处理工具包，在NLP领域中， 最常使用的一个Python库。 
```
conda install nltk
```

##分词工具包

### StandFord NLP

Stanford NLP提供了一系列自然语言分析工具。它能够给出基本的 词形，词性，不管是公司名还是人名等，格式化的日期，时间，量词， 并且能够标记句子的结构，语法形式和字词依赖，指明那些名字指向同 样的实体，指明情绪，提取发言中的开放关系等。
1.一个集成的语言分析工具集；
2.进行快速，可靠的任意文本分析；
3.整体的高质量的文本分析;
4.支持多种主流语言; 
5.多种编程语言的易用接口;
6.方便的简单的部署web服务。

#### 安装

```
Python 版本stanford nlp 安装
• 1)安装stanford nlp自然语言处理包: pip install stanfordcorenlp
• 2)下载Stanford CoreNLP文件

https://stanfordnlp.github.io/CoreNLP/

• 3)下载中文模型jar包
https://nlp.stanford.edu/software/stanford-chinese-corenlp-2018-10-05-models.jar
• 4)把下载的stanford-chinese-corenlp-2018-10-05-models.jar
放在解压后的Stanford CoreNLP文件夹中，改Stanford CoreNLP文件夹名为stanfordnlp（可选）
• 5)在Python中引用模型:
• from stanfordcorenlp import StanfordCoreNLP
• nlp = StanfordCoreNLP(r‘path', lang='zh')
例如：
nlp = StanfordCoreNLP(r'/Users/gonggugu/Desktop/NLP/stanfordnlp', lang='zh')
```


#### 注意事项

端口选择时访问受限
方法： sudo -s
之后运行测试代码时，代码一直没有执行，强行终止发现代码一直在循环time.sleep(1)。
找到源文件/anaconda3/envs/deeplearning/lib/python3.7/site-packages/stanfordcorenlp/corenlp.py 注释掉118行代码，根据网上同样的问题我也强行注释掉了端口选择，注释掉端口检查，然后手动定义端口。


应该是部署web服务是用到的服务端口，我的电脑当时应该是占用了默认的端口，所以程序一直一直等待，注释掉就好了。


### hanlp

HanLP是由一系列模型与算法组成的Java工具包，目标是普及自然 语言处理在生产环境中的应用。HanLP具备功能完善、性能高效、架构 清晰、语料时新、可自定义的特点。      

功能：中文分词 词性标注 命名实体识别 依存句法分析 关键词提取 新词发现 短语提取 自动摘要 文本分类 拼音简繁 

#### Hanlp环境安装
~~• 1、安装Java:~~
~~• 2、安裝Jpype,~~
> conda install -c conda-forge jpype1
>[或者]pip install jpype1

~~• 3、测试是否按照成功:~~
```
from jpype import *
startJVM(getDefaultJVMPath(), "-ea")
java.lang.System.out.println("Hello World")
shutdownJVM()
```


HanLP更新了pyhanlp，支持pip安装和python直接调用。
```

conda install -c conda-forge jpype1
pip install pyhanlp

自动配置：默认在首次调用pyhanlp时自动下载jar包和数据包，并自动完成配置（但是速度尼玛 太慢）

手动配置：则需要按照链接进行操作

https://github.com/hankcs/pyhanlp/wiki/%E6%89%8B%E5%8A%A8%E9%85%8D%E7%BD%AE

下载data-for-1.7.0.zip和hanlp-1.7.0.-release.zip，解压后分别获得data文件夹和hanlp.properties、hanlp-1.7.0.jar文件。

进入python包的安装目录，以Anaconda为例，进入该安装目录下的./lib/site-packages/pyhanlp文件夹，将得到的文件复制到本目录，保证目录的内部结构如下：

hanlp
|—static
|　　|—data
|　　|　　|—dictionary
|　　|　　|—model
|　　|　　|—READ.html
|　　|　　|—version.txt
|　　|—init.py
|　　|—hanlp.properties
|　　|—hanlp.properties.in
|　　|—hanlp-1.7.0.jar
|　　|—index.html
|—init.py
|—main.py
|—server.py
|—util.py

编辑hanlp.properties中的内容，更改其中root=行的内容，使其指向data文件夹的上一层，如root=D:/Anaconda/Lib/site-packages/pyhanlp/static

打开命令行，输入hanlp -v，检查是否安装成功。正常应该返回jar包、data文件夹和hanlp.properties的位置。
```




