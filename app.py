import io
import json
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

model = load_model('model.h5',compile=False)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
st='none'
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    img = request.files.get('image').read()
    img = Image.open(io.BytesIO(img)).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    images = np.vstack([img])
    classes = model.predict(images, batch_size=32)
    res = round(classes[0][0],0)
    if (res == 0):
        return "Defective"
    else:
        return "Non-Defective"

if __name__ == '__main__':
    app.run(debug=True)
