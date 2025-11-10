import io
import os
import base64
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

MODEL_PATH = os.environ.get("", r"C:\Users\msrih\Downloads\handwriting-web\model.h5")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH}' not found. Place your Model.h5 in the project root."
    )

model = tf.keras.models.load_model(MODEL_PATH)
# --------- Helpers ---------
def num_to_sym(x: int) -> str:
    if x == 10:
        return "+"
    if x == 11:
        return "-"
    if x == 12:
        return "*"
    if x == 13:
        return "/"
    if x == 14:
        return "("
    if x == 15:
        return ")"
    if x == 16:
        return "."
    return str(x)

def testing(img: np.ndarray) -> np.ndarray:
    # img expected grayscale 28x28 (or 32x32 incl padding before resizing)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1).astype("float32") / 255.0
    return model.predict(img, verbose=0)

def solve_expression(tokens):
    # tokens: list of tuples (symbol, confidence)
    expr = "".join(sym for sym, _ in tokens)
    try:
        # Evaluate only if expression is composed of allowed characters
        allowed = set("0123456789+-*/(). ")
        if not set(expr).issubset(allowed):
            raise ValueError("Invalid characters in expression")
        # Python eval for arithmetic (no names allowed)
        result = eval(expr, {"__builtins__": None}, {})
        try:
            result = float(f"{float(result):.4f}")
        except Exception:
            pass
        return expr, str(result), None
    except Exception as e:
        return expr, None, str(e)

def annotate_and_predict(bgr_image: np.ndarray):
    # Add small padding around image
    pad = 5
    h, w = bgr_image.shape[:2]
    padded = ~np.ones((h + pad * 2, w + pad * 2, 3), dtype=np.uint8)
    padded[pad:pad + h, pad:pad + w] = bgr_image[:]
    img = padded

    # Slight blur to reduce noise
    img_blur = cv2.GaussianBlur(img, (5, 5), 5)

    # Grayscale and threshold
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

    # Find external contours on inverted image
    inv = cv2.bitwise_not(bw)
    cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Sort left-to-right (similar to original lambda)
    cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2])

    green = (0, 230, 0)
    blue = (225, 0, 0)
    red = (0, 0, 225)

    tokens = []
    annotated = img.copy()

    i = 0
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        i += 1
        cropped = gray[y:y + h, x:x + w]

        # Case for '1' (tall): add left/right padding
        if abs(h) > 1.25 * abs(w) and w > 0:
            padLR = 3 * int(max(1, (h // max(1, w)) ** 3))
            cropped = cv2.copyMakeBorder(cropped, 0, 0, padLR, padLR, cv2.BORDER_CONSTANT, value=255)

        # Case for '-' (flat): add top/bottom padding
        if abs(w) > 1.1 * abs(h) and h > 0:
            padTB = 3 * int(max(1, (w // max(1, h)) ** 3))
            cropped = cv2.copyMakeBorder(cropped, padTB, padTB, 0, 0, cv2.BORDER_CONSTANT, value=255)

        resized = cv2.resize(cropped, (28, 28))
        padded_digit = cv2.copyMakeBorder(resized, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255)

        # Predict
        pred = testing(padded_digit)
        ind = int(np.argmax(pred[0]))
        acc = float(f"{float(pred[0][ind]) * 100:.2f}")
        sym = num_to_sym(ind)
        tokens.append((sym, acc))

        # Draw annotation
        cv2.rectangle(annotated, (x, y), (x + w, y + h), green, 3)
        yim = y + h + 35 if y < 60 else y - 10
        cv2.putText(annotated, f"{sym}", (x, yim), cv2.FONT_HERSHEY_SIMPLEX, 1.2, blue, 3)
        cv2.putText(annotated, f"{acc}%", (x + 30, yim + 28), cv2.FONT_HERSHEY_DUPLEX, 0.8, red, 2)

    # Encode annotated image as base64
    _, buf = cv2.imencode(".png", annotated)
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"
    return tokens, data_url

# --------- Flask App ---------
app = Flask(__name__)

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    data_url = data.get("image")
    if not data_url or not data_url.startswith("data:image"):
        return jsonify({"error": "No image provided as data URL"}), 400

    # Decode base64 image
    try:
        header, b64data = data_url.split(",", 1)
        raw = base64.b64decode(b64data)
        img_array = np.frombuffer(raw, np.uint8)
        bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    tokens, annotated_url = annotate_and_predict(bgr)

    if not tokens:
        return jsonify({"error": "No handwriting detected"}), 200

    expr, result, err = solve_expression(tokens)

    return jsonify({
        "tokens": [{"symbol": s, "confidence": c} for (s, c) in tokens],
        "expression": expr,
        "result": result,
        "error": err,
        "annotated_image": annotated_url
    })

if __name__ == "__main__":
    # Local dev server
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
