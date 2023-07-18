

import io
import requests
import streamlit as st
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
import base64

# Interact with FastAPI endpoint
backend = "http://127.0.0.1:8000/calorie_estimation/"


def process(input_image, server_url: str):
    try:
        m = MultipartEncoder(fields={"file": ("filename", input_image, "image/jpeg")})

        r = requests.post(
            server_url,
            data=m,
            headers={"Content-Type": m.content_type}
            
        )

        # Ensure the request was successful
        if r.status_code == 200:
            # Convert the JSON response to a Python dictionary
            json_response = r.json()

            # Extract image data and convert back to image format
            image_data = base64.b64decode(json_response.get("image", ''))
            image = Image.open(io.BytesIO(image_data))

            # Extract total calories from the response
            total_calories = json_response.get("total_calories", None)

            return image, total_calories

        else:
            raise Exception("Request failed with status code: " + str(r.status_code))

    except Exception as e:
        raise Exception("Error occurred during processing: " + str(e))


# Construct UI layout
st.title("Calories Estimation")

st.write(
    """This object detection project takes in an image and outputs the image with bounding boxes created around the objects in the image and also total calories.
       Visit this URL at `:8000/docs` for FastAPI documentation."""
)

input_image = st.file_uploader("Insert image")  # Image upload widget

if st.button("Get Calories"):
    if input_image:
        try:
            processed_image = Image.open(input_image).convert("RGB")
            st.image(processed_image, caption='Original Image')

            processed_image, total_calories = process(input_image, backend)

            if processed_image is not None and total_calories is not None:
                st.image(processed_image, caption='Processed Image')

                if st.checkbox("Show Total Calories (Cal)"):
                    st.write(total_calories)
            else:
                st.write("Error occurred during processing.")

        except Exception as e:
            st.write("Error occurred: " + str(e))
    else:
        st.write("Insert an image!")
