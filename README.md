# SentimentAnalysis_api
使用django对情感分析功能进行封装，里面包含使用情感词典和Bert模型进行情感分类，最后可以使用tensorFlow serving将模型部署在docker中运行。

### 原理说明
  本项目目的在于将Bert模型和dict词典查询封装为api接口供其他项目调用，感兴趣的人可以进一步加上Restful framework。其中训练好的Bert模型我使用tensorFlow
serving部署在docker中直接启动。原先的训练代码放在目录文件夹下面。

### 环境配置
  ```
    python 3.6.0
    tensorflow 1.10.0以上
  ```
