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
### 代码目录
  整个url的路径配置放在sentiment_api/urls.py中了，这里使用“http://xxxx.xxxx/$ports/sentiment/emotion/”<br/>
  其中的前面的http地址是你的服务器上的地址，端口号是你自己定义使用的<br/>
  
  ```
    sentiment_bert 这个目录下面存放了Bert模型的训练代码，训练好模型后可以自行取出
      Bert 模型训练代码，目录下面省略了中文预训练模型chinese_L-12_H-768_A-12,需要自行下载
      f_dict 情感词典方法所用到的文件
      multiModel 将训练好的模型放在此目录下面
        models.config 为tensorflow serving配置好多模型启动文件
      views.py 我们所用到的逻辑处理文件
      classifiers.py 我们所用到的情感词典类
      sa_predict_saved_model.py 用来生成可以应用于serving模型格式的文件
    sentiment_dic 这个目录下存放的情感词典方法，在sentiment_bert已经有备份
  ```
  
### 部署方法
  代码git下来安装好所有依赖包后<br/>
  python manage.py runserver 0.0.0.0:$ports 这里的0.0.0.0指的是任意都可访问，大家可自行定义使用<br/>
  如果只是学习调用，可以直接加上nohup放在后台运行，如果是部署生产，建议打包成docker镜像<br/>
  训练好的模型取出后，我们可以在docker中安装tensorflow serving部署启动。<br/>
  参考博客https://www.jianshu.com/p/d673c9507988<br/>
  环境配置好了以后<br/>
  运行<br/>
  docker run -p 8501:8501 --mount type=bind,source=/home/yeweiyang/tmp/sentiment_api/sentiment_bert/multiModel/,target=/models/multiModel \
 -t tensorflow/serving --model_config_file=/models/multiModel/models.config<br/>
  实现多模型部署，访问api是http://XXXX.XXXX.XX/v1/models/triple_model:predict<br/>，具体参考models.config。<br/>
  在这里我训练了两个模型triple_model和polarity_model<br/>
  注意：直接从bert中训练的模型需要经过修改才能适应serving，修改文件定义在sa_predict_saved_model.py文件中
  
