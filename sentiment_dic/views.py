# coding:utf-8
from django.http import HttpResponse
import json

# 引入我们创建的表单类
from sentiment_bert.classifiers import DictClassifier

def index(request):
    # if request.method == 'POST':  # 当提交表单时
    #
    #     form = AddForm(request.POST)  # form 包含提交的数据
    #
    #     if form.is_valid():  # 如果提交的数据合法
    #         sentiment = form.cleaned_data['sentiment']
    #
    #         ds = DictClassifier()
    #         dic_value = ds.analyse_sentence(str(sentiment))
    #
    #         return HttpResponse(str(dic_value))
    #
    # else:  # 当正常访问时
    #
    #     form = AddForm()
    # return render(request, 'index.html', {'form': form})

    if request.method == 'POST':
        ds = DictClassifier()
        emotion_text = request.POST.get('message', 0)
        emotion_text = json.loads(emotion_text)
        dic_value = []
        for raw_data in emotion_text:
            dic_value.append(ds.analyse_sentence(raw_data))
        return HttpResponse(str(dic_value))
    else:
        print("abc")
