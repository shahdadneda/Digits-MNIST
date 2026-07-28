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

## Invalid-drawing rejection

The API can abstain instead of forcing every input into one of the ten digit
classes. It rejects drawings that fill or span almost the entire canvas,
or contain several unrelated marks. Rejected responses have
`"accepted": false` and do not expose a misleading digit or confidence score.
Drawings that pass those structural checks always show the model's closest
digit match, even when its confidence is below 80%.

This input gate handles obvious scribbles, but softmax confidence alone cannot
prove that an unfamiliar image is a digit. For a production-grade open-set
recognizer, train an explicit `not_a_digit` class with representative negative
examples (scribbles, shapes, letters, blank/noisy images) and tune its rejection
threshold on a separate validation set.
