# mindspore.nn.GroupNorm
GroupNorm是深度学习中常用的归一化方式。他的作用是在神经网络训练中对输入数据进行归一化，他将输入归一化到均值为0和方差为1的分布中，来防止梯度消失和爆炸，并提高模型的泛化能力。   
GroupNorm的提出用于解决BatchNorm归一化方式对batch size依赖的影响。采用了把channel分成组然后对每一组做归一化处理，不受batch size的约束。
```python
class mindspore.nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True, gamma_init='ones', beta_init='zeros', dtype=mstype.float32)
```
## 输入和输出
输入的Tensor尺寸为（N，C，H，W）。
其中N为批尺寸，C为通道数，H和W分别为高度和宽度。   
输出为归一化后的的Tensor，具有与输入相同的shape和数据类型。      
计算公式为：   
<img src="./formula.jpg" height="80px">    

## 参数
num_groups (int) - 沿通道维度待划分的组数。   
num_channels (int) - 输入的通道数。   
eps (float，可选) - 添加到分母中的值，以确保数值稳定及分母不为0。默认值： 1e-05 。   
affine (bool，可选) - Bool类型，当设置为True时，给该层添加可学习的仿射变换参数，即gama与beta。默认值： True。   
gamma_init (Union[Tensor, str, Initializer, numbers.Number]，可选) - gamma参数的初始化方法。str的值引用自函数 mindspore.common.initializer ，包括 'zeros' 、 'ones' 、 'xavier_uniform' 、 'he_uniform' 等。默认值： 'ones' 。如果gamma_init是Tensor，则shape必须为。     
beta_init (Union[Tensor, str, Initializer, numbers.Number]，可选) - beta参数的初始化方法。str的值引用自函数 mindspore.common.initializer ，包括 'zeros' 、 'ones' 、 'xavier_uniform' 、 'he_uniform' 等。默认值： 'zeros' 如果gamma_init是Tensor，则shape必须为。   
dtype (mindspore.dtype，可选) - Parameters的dtype。默认值： mstype.float32。   

## 与torch.nn.GroupNorm的区别
torch.nn.GroupNorm不能设置gamma_init和beta_init参数，当affine值为True时，gamma为1，beta为0。

## 常见的归一化方式对比：
下图展示了四种常见的归一化方式，其中N为batch size, C为channel, H和W为height和width。
<img src="./4type_normalization.png" height="250px">     

BatchNorm：batch方向做归一化，算N * H * W的均值和方差。      
LayerNorm：channel方向做归一化，算C * H * W的均值和方差。详见[mindspore.nn.LayerNorm接口实践](./../nn.LayerNorm/mindspore.nn.LayerNorm.md)   
InstanceNorm：一个channel内做归一化，算H * W的均值和方差。   
GroupNorm：将channel方向分group，然后每个group内做归一化，算(C/G) * H * W的均值和方差。   

## 样例
输入为batch size为2，channel为4，height为2，width为2的tensor。
[[[[1,0], [0,2]],   
  [[3,4], [1,2]],   
  [[-2,9], [7,5]],   
  [[2,3], [4,2]]],   
  
  [[[1,2], [-1,0]],   
  [[1,2], [3,5]],   
  [[4,7], [-6,4]],   
  [[1,4], [1,5]]]]   

我们设置mindspore.nn.GroupNorm的num_groups为2，num_channels为4。则每组的channel数为 4/2 = 2。应该在下述范围内做归一化处理：   

Batch1_Group1: [[1,0], [0,2]], [[3,4], [1,2]]  均值：1.625 方差: 1.7343     
Batch1_Group2: [[-2,9], [7,5]], [[2,3], [4,2]] 均值：3.75  方差：9.9375    
Batch2_Group3: [[1,2], [-1,0]], [[1,2], [3,5]] 均值：1.625 方差：2.9843      
Batch2_Group4: [[4,7], [-6,4]], [[1,4], [1,5]] 均值：2.5   方差：13.75   
代入计算公式后，可得出与如下代码相近的答案（mindspore.nn.GroupNorm的output还受affine影响）。

```python 
import mindspore as ms
from mindspore import Tensor
import numpy as np

x =  Tensor([[[[1,0], [0,2]],
              [[3,4], [1,2]],
              [[-2,9], [7,5]],
              [[2,3], [4,2]]],
              
             [[[1,2], [-1,0]],
              [[1,2], [3,5]],
              [[4,7], [-6,4]],
              [[1,4], [1,5]]]], ms.float32)
group_norm_op = ms.nn.GroupNorm(2, 4)
output = group_norm_op(x)
print(output)

# output：
# [[[[-0.4745776  -1.2339017 ]
#    [-1.2339017   0.28474656]]

#   [[ 1.0440707   1.8033949 ]
#    [-0.4745776   0.28474656]]

#   [[-1.824018    1.6654078 ]
#    [ 1.0309666   0.39652565]]

#   [[-0.5551359  -0.2379154 ]
#    [ 0.07930513 -0.5551359 ]]]


#  [[[-0.3617867   0.21707201]
#    [-1.5195041  -0.9406454 ]]

#   [[-0.3617867   0.21707201]
#    [ 0.79593074  1.9536481 ]]

#   [[ 0.4045198   1.2135594 ]
#    [-2.2922788   0.4045198 ]]

#   [[-0.4045198   0.4045198 ]
#    [-0.4045198   0.67419964]]]]
```