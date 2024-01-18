import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from io import BytesIO
import sys
import math
import numpy as np
from PIL import Image
from scipy import signal
from sklearn.cluster import KMeans

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maximum file size: 16 MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def estimate_noise(I):
    H, W = I.shape
    M = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
    sigma = np.sum(np.sum(np.absolute(signal.convolve2d(I, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))
    return sigma

def detect_manipulation(image_path, blockSize=32, threshold=0.4):
    im = Image.open(image_path)
    im = im.convert('L')  
    blocks = []
    imgwidth, imgheight = im.size

    for i in range(0, imgheight, blockSize):
        for j in range(0, imgwidth, blockSize):
            box = (j, i, j + blockSize, i + blockSize)
            b = im.crop(box)
            a = np.asarray(b).astype(int)
            blocks.append(a)

    variances = []
    for block in blocks:
        variances.append([estimate_noise(block)])

    kmeans = KMeans(n_clusters=2, random_state=0).fit(variances)
    center1, center2 = kmeans.cluster_centers_
    detected_blocks = [1 if abs(var[0] - center1[0]) > abs(var[0] - center2[0]) else 0 for var in variances]

    fake_percentage = sum(detected_blocks) / len(detected_blocks)
    is_fake = fake_percentage > threshold

 
    im_array = np.array(im)
    for idx, block in enumerate(blocks):
        i, j = divmod(idx, int(imgwidth / blockSize))
        if detected_blocks[idx] == 1:
            im_array[i * blockSize:(i + 1) * blockSize, j * blockSize:(j + 1) * blockSize] = 255  

    
    output_img = Image.fromarray(im_array)
    output_bytes_io = BytesIO()
    output_img.save(output_bytes_io, format='PNG')
    output_bytes_io.seek(0)

    return output_bytes_io, is_fake

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

    
        output_bytes_io, is_fake = detect_manipulation(file_path)

        response = {
            'is_fake': is_fake,
            'image': output_bytes_io.getvalue().decode('latin1') 
        }

        return jsonify(response)

    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
