from django.db import models
from django.contrib.auth.models import User

class PredictionHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    image = models.ImageField(upload_to='uploads/')
    predicted_class = models.CharField(max_length=50)
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user} - {self.predicted_class} ({self.confidence:.2f}%)"