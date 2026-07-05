

import base64
import io
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms
import timm

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "deepfake_model.pth"
IMG_SIZE = 224

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

class HybridDeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientnet = timm.create_model(
            "efficientnet_b0", pretrained=False, num_classes=0, global_pool="avg"
        )
        self.spatial_to_sequence = nn.Linear(1280, 1280)
        self.bilstm = nn.LSTM(
            input_size=64, hidden_size=128,
            num_layers=2, batch_first=True, bidirectional=True,
        )
        self.freq_branch = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(320, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        spatial = self.efficientnet(x)
        spatial = self.spatial_to_sequence(spatial)
        seq = spatial.view(spatial.size(0), 20, 64)
        lstm_out, _ = self.bilstm(seq)
        lstm_pooled = lstm_out.mean(dim=1)
        freq = self.freq_branch(x).view(x.size(0), -1)
        combined = torch.cat([lstm_pooled, freq], dim=1)
        return self.classifier(combined)


def load_model():
    model = HybridDeepfakeModel()
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        target = list(model.efficientnet.blocks.children())[-1]

        def fwd(m, i, o): self.activations = o.detach()
        def bwd(m, gi, go): self.gradients = go[0].detach()

        target.register_forward_hook(fwd)
        target.register_full_backward_hook(bwd)

    def generate(self, tensor):
        self.model.zero_grad()
        out = self.model(tensor)
        out[0, 0].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = F.relu(torch.tensor(cam)).numpy() if not isinstance(cam, np.ndarray) else cam
        if isinstance(cam, torch.Tensor):
            cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def overlay_heatmap(pil_img, cam, alpha=0.45):
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]
    cam_r = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = np.clip((1 - alpha) * img + alpha * heatmap, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

if not os.path.exists(MODEL_PATH):
    model = None
    print(f"⚠️  WARNING: {MODEL_PATH} not found. Place it next to app.py.")
else:
    model = load_model()
    print(f"✅ Model loaded on {DEVICE}")

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model file not found. Add deepfake_model.pth next to app.py."}), 500

    file = request.files.get("image")
    threshold = float(request.form.get("threshold", 0.5))

    if not file:
        return jsonify({"error": "No image uploaded."}), 400

    image = Image.open(file.stream).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        fake_prob = float(model(tensor)[0, 0].cpu())

    fake_conf = round(fake_prob * 100, 2)
    real_conf = round((1 - fake_prob) * 100, 2)
    verdict = "FAKE" if fake_prob >= threshold else "REAL"

    try:
        gc = GradCAM(model)
        cam = gc.generate(tensor)
        heatmap = overlay_heatmap(image, cam)
        heatmap_b64 = pil_to_b64(heatmap)
    except Exception:
        heatmap_b64 = pil_to_b64(image)  

    original_b64 = pil_to_b64(image)

    return jsonify({
        "verdict": verdict,
        "fake_conf": fake_conf,
        "real_conf": real_conf,
        "heatmap": heatmap_b64,
        "original": original_b64,
        "device": str(DEVICE),
    })


if __name__ == "__main__":
    app.run(debug=True)
