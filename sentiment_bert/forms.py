from django import forms


class AddForm(forms.Form):
    sentiment = forms.CharField(max_length=120,label='名称')