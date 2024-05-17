from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
import json
from .modules.predict import Model


def index(request):
    template = loader.get_template('ml/index.html')
    return HttpResponse(template.render(None, request))

def predict(request):
  if request.method == 'GET':
    try:
        data = request.GET.get('text', '')
        model = Model(data)
        model.vectorize()
        result = model.predict()
        return HttpResponse(json.dumps({"rating": result}))
    except KeyError:
        return HttpResponse("error")