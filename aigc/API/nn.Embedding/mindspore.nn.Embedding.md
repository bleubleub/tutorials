# mindspore.nn.Embedding
mindspore.nn.Embedding用于将词汇表中的单词或标记映射到固定大小的向量空间。这个向量空间通常被称为词向量空间，每个单词或标记在这个空间中都有一个唯一的向量表示。Embedding层为神经网络提供了一种将文本数据转换为可处理数值形式的方法。

```python
class mindspore.nn.Embedding(vocab_size, embedding_size, use_one_hot=False, embedding_table='normal', dtype=mstype.float32, padding_idx=None)
```

## 输入和输出：
输入的Tensor大小为（batch_size, x_length）, 输入的元素为整形，且元素数目必须小于等于vocab_size。
输出的Tensor大小为（batch_size, x_length, embedding_size）。

## 参数：
**vocab_size** (int) - 词典的大小。   
**embedding_size** (int) - 每个嵌入向量的大小。   
**use_one_hot** (bool) - 指定是否使用one-hot形式。默认值： False 。   
**embedding_table** (Union[Tensor, str, Initializer, numbers.Number]) - embedding_table的初始化方法。当指定为字符串，字符串取值请参见类 mindspore.common.initializer 。默认值： "normal" 。   
**dtype** (mindspore.dtype) - x的数据类型。默认值： mstype.float32 。   
**padding_idx** (int, None) - 将 padding_idx 对应索引所输出的嵌入向量用零填充。默认值： None 。该功能已停用。   

## 样例
以下述代码举例，构造一个batch_size(b)为2、x_length(x)为4的输入：   
[[1, 2, 4, 5],   
 [4, 3, 2, 9]]   
定义一个vocab_size(v)为10，embedding_size(e)为3的词向量表(embedding_table)，并使用one-hot形式：   
net = nn.Embedding(10, 3,  True)   
one-hot形式会先将输入改写为一个大小为[2, 4, 10]的Tensor, 其中2和4为原输入的b和x, 10为embedding_table的v。
然后再将这个[2,4,10]的Tensor与embedding_table(大小为[10,3])做矩阵乘法，即可以得到一个输出为[2,4,3]的Tensor。

```python
import mindspore
from mindspore import Tensor, nn
import numpy as np

net = nn.Embedding(10, 3,  True)
x = Tensor([[1, 2, 4, 5],
            [4, 3, 2, 9]], mindspore.int32)
# Maps the input word IDs to word embedding.
output = net(x)
result = output.shape
print(net.embedding_table.shape)
# (10, 3)
print(net.embedding_table.asnumpy())
# [[-0.00575683 -0.0090051  -0.0020029 ]
#  [ 0.00470815 -0.00325267 -0.01868797]
#  [-0.00202428  0.00765565 -0.00511093]
#  [ 0.01192129  0.00691468 -0.00871143]
#  [-0.01576921  0.0033025  -0.01432611]
#  [ 0.01114684 -0.00698112 -0.00035679]
#  [ 0.00159931  0.0012864  -0.0039245 ]
#  [ 0.0023317  -0.02182413  0.00862475]
#  [ 0.00192053  0.00812579  0.00712475]
#  [ 0.00306729  0.01914453  0.00779424]]
print(result)
# (2, 4, 3)
print(output)
# [[[ 0.00470815 -0.00325267 -0.01868797]
#   [-0.00202428  0.00765565 -0.00511093]
#   [-0.01576921  0.0033025  -0.01432611]
#   [ 0.01114684 -0.00698112 -0.00035679]]

#  [[-0.01576921  0.0033025  -0.01432611]
#   [ 0.01192129  0.00691468 -0.00871143]
#   [-0.00202428  0.00765565 -0.00511093]
#   [ 0.00306729  0.01914453  0.00779424]]]
```