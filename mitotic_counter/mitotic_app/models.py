# models.py
from django.db import models
import os

class Analysis(models.Model):
    uploaded_image = models.ImageField(upload_to='uploads/')
    video_file = models.FileField(upload_to='videos/', null=True, blank=True)
    processed_video = models.FileField(upload_to='videos/processed/', null=True, blank=True)
    upload_date = models.DateTimeField(auto_now_add=True)
    
    # HPF-related fields
    x_mpp = models.FloatField(null=True, blank=True)
    y_mpp = models.FloatField(null=True, blank=True)
    hpf_width_px = models.IntegerField(null=True, blank=True)
    hpf_height_px = models.IntegerField(null=True, blank=True)
    total_hpfs = models.IntegerField(null=True, blank=True)
    mitoses_per_10_hpf = models.FloatField(null=True, blank=True)
    tumor_grade = models.IntegerField(null=True, blank=True)
    
    def __str__(self):
        return f"Analysis {self.id} - {self.upload_date.strftime('%Y-%m-%d %H:%M')}"
    
    def update_hpf_analysis(self):
        """Update HPF analysis based on current mitotic count"""
        from .utils.hpf_calculator import get_tumor_grade, mitoses_per_10_hpf
        
        # Get current mitotic count
        mitotic_count = self.figures.filter(category='mitotic').count()
        
        # If we have HPF data, recalculate mitoses_per_10_hpf
        if self.total_hpfs:
            self.mitoses_per_10_hpf = mitoses_per_10_hpf(mitotic_count, self.total_hpfs)
            self.mitoses_per_10_hpf = round(self.mitoses_per_10_hpf, 2)
            self.tumor_grade = get_tumor_grade(self.mitoses_per_10_hpf)
            self.save()
            
            # Print for debugging
            print(f"Updated HPF analysis: {mitotic_count} mitotic figures, {self.mitoses_per_10_hpf} per 10 HPF, Grade {self.tumor_grade}")
    
class DetectedFigure(models.Model):
    MITOTIC = 'mitotic'
    NON_MITOTIC = 'non_mitotic'
    DISCARDED = 'discarded'
    
    CATEGORY_CHOICES = [
        (MITOTIC, 'Mitotic'),
        (NON_MITOTIC, 'Non-Mitotic'),
        (DISCARDED, 'Discarded'),
    ]
    
    analysis = models.ForeignKey(Analysis, on_delete=models.CASCADE, related_name='figures')
    image_file = models.ImageField(upload_to='figures/')
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    confidence = models.FloatField()
    frame_number = models.IntegerField()
    
    def __str__(self):
        return f"{self.category} figure ({self.confidence:.2f}) - Frame {self.frame_number}"
    
    def filename(self):
        return os.path.basename(self.image_file.name)
    
    def save(self, *args, **kwargs):
        # Save the figure
        super().save(*args, **kwargs)
        
        # Update HPF analysis on the parent analysis object
        if self.analysis:
            self.analysis.update_hpf_analysis()