# 原文说明
该代码为文章《面向鸟鸣声识别任务的深度学习技术研究》的实现


文章链接：https://www.biodiversity-science.net/CN/10.17520/biods.2022308


使用该代码请添加以下引用：


谢卓钒, 李鼎昭, 孙海信, 张安民(2023)面向鸟鸣声识别任务的深度学习技术. 生物多样性, 31, 22308. doi: 10.17520/biods.2022308. 
Xie ZF, Li DZ, Sun HX, Zhang AM (2023)Deep learning techniques for bird chirp recognition task. Biodiversity Science, 31, 22308. doi: 10.17520/biods.2022308. 


# 代码说明
该任务融合了CBAM与DenseNet121，使用中心损失函数及鸟声融合特征进行数据预处理、鸟声识别等任务。


所使用的鸟声数据均来自于Xeno-Canto网站，由于数据过大无法上传，如有需要可以联系邮箱haloxxie@stu.xmu.edu.cn


注：原始数据中存在数据不平衡问题，所以我们通过一些数据增强方法对数据做了预处理，但受限于篇幅没有在原文中体现，并且代码中使用的是处理过后的npy文件。
