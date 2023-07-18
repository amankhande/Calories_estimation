import io
from prediction import*
from starlette.responses import Response
from fastapi import FastAPI, File
import uvicorn
from fastapi import FastAPI, UploadFile, File
from starlette.responses import JSONResponse
import tempfile
import base64
import os
from fastapi import FastAPI, UploadFile, HTTPException
from PIL import Image


app = FastAPI(
    title="Calories Estimation",
    description="""This object detection project takes in an image and outputs the image with bounding boxes created around the objects in the image and also total calories""",
    version="0.1.0",
)







@app.post("/calorie_estimation/")
async def calorie_estimation(image: UploadFile = File(...)):
    try:
        temp_folder = "temp/"
        os.makedirs(temp_folder, exist_ok=True)

        file_location = os.path.join(temp_folder, image.filename)
        with open(file_location, "wb+") as file_object:
            file_object.write(await image.read())
        file_path = f"{temp_folder}{image.filename}"
        image, total_calories = prediction(file_path)

        # Convert numpy image into PIL image
        image_pil = Image.fromarray(image.astype('uint8'))

        bytes_io = io.BytesIO()
        image_pil.save(bytes_io, format="PNG")
        encoded_image = base64.b64encode(bytes_io.getvalue()).decode("utf-8")

        return JSONResponse(status_code=200, content={"image": encoded_image, "total_calories": total_calories})

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image Upload Failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    pip freeze > requirements.txt
