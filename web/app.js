const canvas = document.querySelector("#drawingCanvas");
const context = canvas.getContext("2d", { willReadFrequently: true });
const canvasFrame = canvas.closest(".canvas-frame");
const clearButton = document.querySelector("#clearButton");
const predictButton = document.querySelector("#predictButton");
const predictButtonText = document.querySelector("#predictButtonText");
const predictionDigit = document.querySelector("#predictionDigit");
const predictionConfidence = document.querySelector("#predictionConfidence");
const topChoicesList = document.querySelector("#topChoicesList");
const resultMessage = document.querySelector("#resultMessage");
const connection = document.querySelector("#connection");
const connectionText = document.querySelector("#connectionText");

let isDrawing = false;
let hasDrawing = false;
let lastPoint = null;

function showChoicePlaceholder() {
  const placeholder = document.createElement("li");
  placeholder.className = "choice-placeholder";
  placeholder.textContent = "Your three strongest matches will appear here.";
  topChoicesList.replaceChildren(placeholder);
}

function showTopChoices(probabilities) {
  const topChoices = Object.entries(probabilities)
    .map(([digit, probability]) => ({
      digit: Number(digit),
      probability: Number(probability),
    }))
    .sort((first, second) => second.probability - first.probability)
    .slice(0, 3);

  const fragment = document.createDocumentFragment();

  topChoices.forEach((choice, index) => {
    const row = document.createElement("li");
    row.className = "choice-row";
    if (index === 0) {
      row.classList.add("is-top-choice");
    }

    const rank = document.createElement("span");
    rank.className = "choice-rank";
    rank.textContent = `#${index + 1}`;

    const digit = document.createElement("span");
    digit.className = "choice-digit";
    digit.textContent = `Digit ${choice.digit}`;

    const confidence = document.createElement("span");
    confidence.className = "choice-value";
    confidence.textContent = `${(choice.probability * 100).toFixed(2)}%`;

    row.append(rank, digit, confidence);
    fragment.append(row);
  });

  topChoicesList.replaceChildren(fragment);
}

function resetResults() {
  predictionDigit.textContent = "—";
  predictionConfidence.textContent = "Waiting for a drawing";
  resultMessage.textContent =
    "Draw a single digit, then ask the model to recognize it.";
  resultMessage.classList.remove("is-error");
  showChoicePlaceholder();
}

function clearCanvas() {
  context.save();
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.restore();
  isDrawing = false;
  hasDrawing = false;
  lastPoint = null;
  canvasFrame.classList.remove("has-drawing");
  resetResults();
}

function canvasPoint(event) {
  const bounds = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - bounds.left) * (canvas.width / bounds.width),
    y: (event.clientY - bounds.top) * (canvas.height / bounds.height),
  };
}

function beginDrawing(event) {
  event.preventDefault();
  isDrawing = true;
  hasDrawing = true;
  lastPoint = canvasPoint(event);
  canvasFrame.classList.add("has-drawing");
  canvas.setPointerCapture(event.pointerId);

  context.beginPath();
  context.arc(lastPoint.x, lastPoint.y, 16, 0, Math.PI * 2);
  context.fillStyle = "#11120f";
  context.fill();
}

function continueDrawing(event) {
  if (!isDrawing || lastPoint === null) {
    return;
  }

  event.preventDefault();
  const nextPoint = canvasPoint(event);
  context.beginPath();
  context.moveTo(lastPoint.x, lastPoint.y);
  context.lineTo(nextPoint.x, nextPoint.y);
  context.strokeStyle = "#11120f";
  context.lineWidth = 32;
  context.lineCap = "round";
  context.lineJoin = "round";
  context.stroke();
  lastPoint = nextPoint;
}

function finishDrawing(event) {
  if (!isDrawing) {
    return;
  }

  isDrawing = false;
  lastPoint = null;
  if (canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
}

function canvasBlob() {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error("The drawing could not be converted into an image."));
      }
    }, "image/png");
  });
}

function showPrediction(data) {
  const confidencePercent = data.confidence * 100;
  predictionDigit.textContent = String(data.digit);
  predictionConfidence.textContent = `${confidencePercent.toFixed(2)}% confidence`;
  resultMessage.textContent =
    "Prediction complete. Clear the canvas to try another digit.";
  resultMessage.classList.remove("is-error");
  showTopChoices(data.probabilities);
}

function showError(message) {
  resultMessage.textContent = message;
  resultMessage.classList.add("is-error");
}

async function predictDrawing() {
  if (!hasDrawing) {
    showError("Draw a digit on the canvas before asking for a prediction.");
    canvas.focus();
    return;
  }

  predictButton.disabled = true;
  clearButton.disabled = true;
  predictButtonText.textContent = "Recognizing…";
  resultMessage.textContent = "Preparing your drawing for the model…";
  resultMessage.classList.remove("is-error");

  try {
    const blob = await canvasBlob();
    const formData = new FormData();
    formData.append("image", blob, "canvas-digit.png");

    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail ?? "The model could not process this drawing.");
    }

    showPrediction(data);
  } catch (error) {
    showError(error.message ?? "The backend could not be reached.");
  } finally {
    predictButton.disabled = false;
    clearButton.disabled = false;
    predictButtonText.textContent = "Recognize digit";
  }
}

async function checkConnection() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error("Health check failed");
    }

    connection.classList.add("is-online");
    connection.classList.remove("is-offline");
    connectionText.textContent = "Model online";
  } catch {
    connection.classList.add("is-offline");
    connection.classList.remove("is-online");
    connectionText.textContent = "Model offline";
  }
}

canvas.addEventListener("pointerdown", beginDrawing);
canvas.addEventListener("pointermove", continueDrawing);
canvas.addEventListener("pointerup", finishDrawing);
canvas.addEventListener("pointercancel", finishDrawing);
clearButton.addEventListener("click", clearCanvas);
predictButton.addEventListener("click", predictDrawing);

canvas.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && hasDrawing) {
    predictDrawing();
  }
});

clearCanvas();
checkConnection();
