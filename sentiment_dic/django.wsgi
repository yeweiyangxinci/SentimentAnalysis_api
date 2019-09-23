#!/usr/local/bin/python

import os
import sys

path='/home/yeweiyang/tmp/' #change to your path.DON'T foget the last'/'

sys.path.append(path)
sys.path.append(path+'sentiment_api')

os.environ['DJANGO_SETTINGS_MODULE'] = 'sentiment_api.settings'

import django.core.handlers.wsgi
application = django.core.handlers.wsgi.WSGIHandler()