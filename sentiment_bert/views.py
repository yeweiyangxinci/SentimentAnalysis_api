# coding:utf-8
from django.http import HttpResponse
import json

# 引入我们创建的表单类

from .Bert.serving_predict import BertClassifier
# 引入我们创建的表单类
from .classifiers import DictClassifier

def index(request):
    if request.method == 'POST':

        ds = DictClassifier()
        bert = BertClassifier()

        result_arr = []
        text_arr = json.loads(request.body)
        for item in text_arr:
            item_id = item.get('id')
            item_text = item.get('msg')
            emotionValue = float(ds.analyse_sentence(item_text))
            emotion = bert.run_server("polarity", item_text)
            if emotion == "neg" and emotionValue<0:
                emotion = bert.run_server("triple", item_text)
            else:
                emotion = "开心"
            dict_item = {'id': item_id, 'emotionValue': emotionValue, 'emotion': emotion}
            result_arr.append(dict_item)
        return HttpResponse(json.dumps(result_arr, ensure_ascii=False))