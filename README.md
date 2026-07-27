# MNIST Handwritten Digit Recognizer

This project trains a small convolutional neural network to recognize the
handwritten digits 0 through 9.

## Set up

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick test

This downloads MNIST and trains on a small subset to verify the complete
training pipeline:

```bash
python train.py --epochs 1 --limit-train 2000 --limit-test 500
```

## Full training

```bash
python train.py --epochs 5
```

The trained weights are saved to `models/mnist_cnn.pt`.

## Predict an MNIST test image

Use an index from 0 through 9999:

```bash
python predict.py --index 42
```

## Predict an ordinary image

The image should contain one clearly visible handwritten digit:

```bash
python predict_image.py path/to/digit.png
```

To also save the exact 28x28 image given to the neural network:

```bash
python predict_image.py path/to/digit.png --preview processed_digit.png
```

The preview command will not overwrite an existing file. Choose a new preview
filename if the requested path already exists.

## Run the local prediction API

Start the FastAPI backend:

```bash
python -m uvicorn api:app --reload
```

The local API is available at `http://127.0.0.1:8000`. Check that it is running:

```bash
curl http://127.0.0.1:8000/health
```

Upload a digit image:

```bash
curl -X POST -F "image=@path/to/digit.png" \
  http://127.0.0.1:8000/predict
```

The backend accepts PNG and JPEG images up to 5 MB. Stop the local backend by
pressing `Control-C` in the terminal where it is running.

The same address also serves the local drawing interface:

```text
http://127.0.0.1:8000
```

Draw one digit, select **Recognize digit**, and the page will upload the canvas
to `/predict` and display the model's three most likely choices.
