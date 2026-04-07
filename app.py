from flask import Flask, render_template, request
import os
from inference import predict_image
from blip_model import generate_caption

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    caption = None
    image_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            result = predict_image(path)
            caption = generate_caption(path)
            image_path = path

    return render_template('index.html',
                           result=result,
                           caption=caption,
                           image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)