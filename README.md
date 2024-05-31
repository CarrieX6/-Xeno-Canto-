# 原文说明

该代码为文章《面向鸟鸣声识别任务的深度学习技术研究》的实现。该任务融合了CBAM与DenseNet121，使用中心损失函数及鸟声融合特征进行数据预处理、鸟声识别等任务。

原文链接：https://www.biodiversity-science.net/CN/10.17520/biods.2022308

# 引用

使用该代码请添加以下引用：

```

@article{:22308,
author = {Zhuofan Xie, Dingzhao Li, Haixin Sun, Anmin Zhang},
title = {Deep learning techniques for bird chirp recognition task},
publisher = {Biodiv Sci},
year = {2023},
journal = {Biodiversity Science},
volume = {31},
number = {1},
eid = {22308},
numpages = {},
pages = {22308},
keywords = {|bird chirp recognition|feature fusion|self-attentive module|central loss function},
url = {https://www.biodiversity-science.net/EN/abstract/article_82770.shtml},
doi = {10.17520/biods.2022308}
}    

```


# 数据说明

所使用的鸟声数据均来自于Xeno-Canto网站，由于数据过大无法上传至Github，如有需要可以通过下方链接下载：

链接：https://pan.quark.cn/s/ef0abf6b45ab

注：原始数据中存在数据不平衡问题，所以我们通过一些数据增强方法对数据做了预处理，但受限于篇幅没有在原文中体现，并且代码中使用的是处理过后的npy文件。
