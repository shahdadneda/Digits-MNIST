"""FastAPI backend that serves predictions from the trained MNIST model."""

from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from image_processing import image_to_tensor
from model import MnistCNN


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg"}
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "mnist_cnn.pt"
WEB_DIR = PROJECT_DIR / "web"


class PredictionResponse(BaseModel):
    digit: int
    confidence: float
    probabilities: dict[str, float]


# These lines run once when the backend starts, not once per request.
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model = MnistCNN()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


app = FastAPI(
    title="MNIST Digit Recognition API",
    description="Upload one handwritten digit and receive the model's prediction.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Confirm that the backend started and loaded the model."""
    return {
        "status": "ok",
        "model": MODEL_PATH.name,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(description="A PNG or JPEG containing one digit."),
) -> PredictionResponse:
    """Validate an uploaded image and return digit probabilities."""
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Upload a PNG or JPEG image.",
        )

    try:
        image_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await image.close()

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image is empty.",
        )
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="The image is larger than the 5 MB upload limit.",
        )

    try:
        with Image.open(BytesIO(image_bytes)) as uploaded_image:
            uploaded_image.load()
            image_tensor, _ = image_to_tensor(
                uploaded_image,
                checkpoint["mnist_mean"],
                checkpoint["mnist_std"],
            )
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The upload is not a readable PNG or JPEG.",
        ) from None
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from None

    with torch.inference_mode():
        logits = model(image_tensor.unsqueeze(0))
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_digit = int(probabilities.argmax().item())
    confidence = float(probabilities[predicted_digit].item())
    probability_map = {
        str(digit): float(probability)
        for digit, probability in enumerate(probabilities.tolist())
    }

    return PredictionResponse(
        digit=predicted_digit,
        confidence=confidence,
        probabilities=probability_map,
    )


# This mount comes last so /health, /predict, and /docs keep their API routes.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
