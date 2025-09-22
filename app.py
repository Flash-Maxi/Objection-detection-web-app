import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from PIL import Image, ImageDraw, ImageFont
import os
import uuid
from flask import Flask, request, render_template

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CONFIDENCE_THRESHOLD = 0.5  # 50% probability

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# --- Load the Model Once ---
print("Loading pre-trained RetinaNet model... This might take a moment.")
weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
model = retinanet_resnet50_fpn_v2(weights=weights)
model.eval()
preprocess = weights.transforms()
class_names = weights.meta["categories"]
print("Model loaded successfully.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_detection(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        predictions = model(img_tensor)

    prediction = predictions[0]

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    detected_objects = []
    for score, label, box in zip(prediction['scores'], prediction['labels'], prediction['boxes']):
        if score > CONFIDENCE_THRESHOLD:
            box = [i.item() for i in box]
            class_name = class_names[label.item()]
            probability = score.item() * 100
            detected_objects.append(f"{class_name}: {probability:.2f}%")

            draw.rectangle(box, outline="lime", width=3)
            text = f"{class_name} {probability:.1f}%"
            try:
                text_bbox = draw.textbbox((box[0], box[1] - 5), text, font=font)
                text_height = text_bbox[3] - text_bbox[1]
                text_width = text_bbox[2] - text_bbox[0]
                draw.rectangle(
                    (box[0], box[1] - text_height - 8, box[0] + text_width + 4, box[1]),
                    fill="lime"
                )
                draw.text((box[0] + 2, box[1] - text_height - 6), text, fill="black", font=font)
            except AttributeError:  # Older Pillow fallback
                draw.text((box[0] + 2, box[1] - 15), text, fill="black")

    unique_filename = f"{uuid.uuid4()}.jpg"
    output_path = os.path.join(app.config['RESULTS_FOLDER'], unique_filename)
    img.save(output_path)
    return unique_filename, detected_objects

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")

        if file and allowed_file(file.filename):
            filename = f"{uuid.uuid4()}.jpg"
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            result_filename, detections = run_detection(upload_path)
            return render_template(
                'index.html',
                result_image=result_filename,
                detections=detections
            )

    return render_template('index.html')

# --- Production Safe Entrypoint ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Respect hosting platform's port
    debug_mode = os.environ.get("FLASK_DEBUG", "1") == "1"  # Allow toggling via env var
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
