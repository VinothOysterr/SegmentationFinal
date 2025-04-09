from fastapi import FastAPI, File, Response, UploadFile
import cv2
import mediapipe as mp
import numpy as np
from fastapi.responses import StreamingResponse
import io
import base64

from pydantic import BaseModel

app = FastAPI()

class SegmentData(BaseModel):
    ImageData: str

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

BG_COLOR = (192, 192, 192)  # Gray background
TRANSPARENT_BG_COLOR = (0, 0, 0) 

@app.post("/segment-image/")
async def segment_image(file: UploadFile = File(...), background: str = "4"):
    # Read the uploaded image
    image_data = await file.read()
    nparray = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparray, cv2.IMREAD_COLOR)

    # Convert BGR to RGB for MediaPipe processing
    results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Prepare output image based on background choice
    if background == '4':
        # Convert image to RGBA (add alpha channel)
        output_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        # Create alpha channel: 255 for human (opaque), 0 for background (transparent)
        alpha_channel = np.where(results.segmentation_mask > 0.8, 255, 0).astype(np.uint8)
        # Set the alpha channel in the output image
        output_image[..., 3] = alpha_channel
    
    # Encode the output image to bytes
    _, img_encoded = cv2.imencode('.png', output_image)
    img_bytes = img_encoded.tobytes()
    
    # Return the segmented image as a streaming response
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")

@app.put("/seg-img/")
async def seg_img(data: SegmentData, background: str = "4"):
    # Decode Base64 to bytes
    image_bytes = base64.b64decode(data.ImageData)
    nparray = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparray, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image data"}

    # Process segmentation (assuming selfie_segmentation is already initialized)
    results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if background == '4':
        # Transparent background
        output_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        alpha_channel = np.where(results.segmentation_mask > 0.8, 255, 0).astype(np.uint8)
        output_image[..., 3] = alpha_channel

    # Encode processed image to PNG
    _, img_encoded = cv2.imencode('.png', output_image)

    # Convert to base64 and return as JSON
    img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
    return {"ImageData": img_base64}

@app.post("/seg-img/")
async def seg_img(file: bytes = File(...), background: str = "4"):
    # Convert bytes to OpenCV image
    nparray = np.frombuffer(file, np.uint8)
    image = cv2.imdecode(nparray, cv2.IMREAD_COLOR)

    if image is None:
        return Response(content="Invalid image data", status_code=400)

    # Convert BGR to RGB for processing
    results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if background == '4':
        # Transparent background
        output_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        alpha_channel = np.where(results.segmentation_mask > 0.8, 255, 0).astype(np.uint8)
        output_image[..., 3] = alpha_channel

    # Encode processed image to PNG
    _, img_encoded = cv2.imencode('.png', output_image)

    return Response(content=img_encoded.tobytes(), media_type="image/png")

# Run the server: uvicorn server:app --reload
if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app",  host="0.0.0.0",port=6542)