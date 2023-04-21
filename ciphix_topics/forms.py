from django import forms

class CommentForm(forms.Form):
    message = forms.CharField(widget=forms.Textarea(attrs={'style' : 'width: 80%; height: 170px; display: block; margin-left: auto; margin-right: auto;'}))