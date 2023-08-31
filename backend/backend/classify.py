# backend/app/classify.py

import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.http import JsonResponse


# model path
model_path = os.path.join(settings.BASE_DIR, 'models', 'test', 'model_0.h5')

# for debugging
print("Model Path:", model_path)

# load model
model = load_model(model_path)


# classification funtion
def classify_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions[0]).numpy()
    return predicted_class

# def classify_image_view(request):
#     if request.method == 'POST' and request.FILES['image']:
#         image = request.FILES['image']
#         fs = FileSystemStorage(location='images')
#         saved_image_path = fs.save(image.name, image)
#         image_path = os.path.join(fs.location, saved_image_path)
#         classification_result = classify_image(image_path)
#         return render(request, 'classify.html', {'result': classification_result})
#     return render(request, 'classify.html')



def classify_image_view(request):
    # check if there is POST request and a picture named 'image' uploaded
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image'] # get the object of the uploaded file
        fs = FileSystemStorage(location='images') # get an object for storing with location
        saved_image_path = fs.save(image.name, image) # save the image in server
        image_path = os.path.join(fs.location, saved_image_path) # get full path for the input
        classification_result = classify_image(image_path) # get the result of classification
        return JsonResponse({'result': str(classification_result)}) # return Json response to the browser
    return render(request, 'classify.html')
