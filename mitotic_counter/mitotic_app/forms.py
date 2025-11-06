# forms.py
from django import forms
from .models import Analysis

class TiffUploadForm(forms.ModelForm):
    class Meta:
        model = Analysis
        fields = ['uploaded_image']
        
    def clean_uploaded_image(self):
        image = self.cleaned_data.get('uploaded_image')
        if image:
            name = image.name.lower()
            if not (name.endswith('.tif') or name.endswith('.tiff')):
                raise forms.ValidationError("Only TIFF files are supported")
        return image