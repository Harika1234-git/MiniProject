from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import PredictionHistory
from django.db.models import Count

import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

MODEL_PATH = "model/inceptionv3_vit_lungcancer.h5"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Benign", "Malignant", "Normal"]

print(" Loading model...")
model = load_model(MODEL_PATH)
print(" Model loaded successfully!")

# Create your views here.
def userhome(request):
    user = request.user
    return render(request, 'User/userhome.html', {'user':user})

@login_required
def prediction(request):
    prediction_result = None

    if request.method == "POST" and request.FILES.get("image"):
        uploaded_file = request.FILES["image"]

        file_path = f"media/uploads/{uploaded_file.name}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb+") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        img = image.load_img(file_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        pred_idx = np.argmax(preds, axis=1)[0]
        confidence = float(preds[0][pred_idx] * 100)
        predicted_class = CLASS_NAMES[pred_idx]

        PredictionHistory.objects.create(
            user=request.user,
            image=uploaded_file,
            predicted_class=predicted_class,
            confidence=confidence
        )

        prediction_result = {
            "class": predicted_class,
            "confidence": f"{confidence:.2f}%"
        }

    return render(request, "User/prediction.html", {"result": prediction_result})

def datavisulization(request):
    return render(request, 'User/datavisulization.html')

def exsisting(request):
    return render(request, 'User/exsisting.html')

def proposed(request):
    return render(request, 'User/proposed.html')

@login_required
def history(request):
    user_history = PredictionHistory.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'User/history.html', {'history': user_history})

@login_required
def analytics(request):
    class_counts = (
        PredictionHistory.objects
        .filter(user=request.user)
        .values('predicted_class')
        .annotate(total=Count('predicted_class'))
    )

    labels = [entry['predicted_class'] for entry in class_counts]
    data = [entry['total'] for entry in class_counts]

    context = {
        'labels': labels,
        'data': data,
    }

    return render(request, 'User/analytics.html', context)