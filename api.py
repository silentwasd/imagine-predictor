import deepdanbooru as dd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, HTTPException

model_path = "./deepdanbooru-v3-20211112-sgd-e28"
model = dd.project.load_model_from_project(model_path, compile_model=False)

tags = dd.project.load_tags_from_project(model_path)

def predict_tags(image: Image.Image, threshold=0.6):
    image = image.convert("RGB")
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    image_array = np.expand_dims(np.array(image), axis=0) / 255.0

    predictions = model.predict(image_array)[0]
    result_tags = result_tags = {tags[i]: float(predictions[i]) for i in range(len(tags)) if predictions[i] > threshold}

    return sorted(result_tags.items(), key=lambda x: x[1], reverse=True)

app = FastAPI()

@app.get("/predict")
def predict_from_url(url: str, threshold: float = 0.6):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image: {str(e)}")
    except IOError:
        raise HTTPException(status_code=400, detail="Invalid image format")

    tags = predict_tags(image, threshold)
    return {"tags": tags}
