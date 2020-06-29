# -*- coding: utf-8 -*-
"""
An app to make inferences from geofignet.
"""
from io import BytesIO, StringIO
import uuid
import base64

import urllib
import requests
from PIL import Image

from flask import Flask
from flask import make_response, send_file
from flask import request, jsonify, render_template, flash

import numpy as np
import torch
from torchvision import models

import geofignet as gfn


CLASS_NAMES = ['blockdiagram',
               'chronostrat',
               'corephoto',
               'correlation',
               'equation',
               'fieldnote',
               'fmilog',
               'geologicmap',
               'outcrop',
               'photomicrograph',
               'regionalsection',
               'rosediagram',
               'seismic',
               'semicrograph',
               'stereonet',
               'structuremap',
               'synthetic',
               'table',
               'ternary',
               'welllog',
              ]


# Instantiate a vanilla ResNet and adjust its shape.
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))

# Load the geofignet weights.
device = torch.device("cpu")
model = torch.load('data/geofignet.pt', map_location=device)

# Evaluate it before inference.
_ = model.eval()


# Error handling
class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

app = Flask(__name__)


# Routes and handlers
@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route('/api', methods=["GET", "POST"])
def api():

    if request.method == "GET":
        if not request.args:
            return render_template('api.html')
        reqs = [{'image': request.args.get('image')}]
    else:
        reqs = request.json.get('requests')

    if len(reqs) > 10:
        return "Number of images exceeded maximum of 10."

    result = {'job_uuid': uuid.uuid1()}
    result['n_requests'] = len(reqs)
    result['results'] = []

    for req in reqs:
        if req['image'].startswith('http'):
            r = requests.get(req['image'])
            img = Image.open(BytesIO(r.content)).convert('RGB')
        else:
            img = Image.open(BytesIO(base64.b64decode(req['image']))).convert('RGB')

        probs = gfn.infer(model, img)

        prob = max(probs)
        clas = CLASS_NAMES[np.argmax(probs)]

        this = {'top_class': clas}
        this['top_prob'] = prob
        this['classes'] = CLASS_NAMES
        this['probabilities'] = probs

        result['results'].append(this)

    return jsonify(result)


@app.route('/infer', methods=["GET"])
def infer():

    url = urllib.parse.unquote(request.args.get('image'))
    if not url:
        flash("You must provide the URL of an image.")
        return render_template('index.html')

    # Get image.
    r = requests.get(url)
    img = Image.open(BytesIO(r.content))

    probs = gfn.infer(model, img)

    prob = format(max(probs), "0.3f")
    clas = CLASS_NAMES[np.argmax(probs)]

    plot = gfn.plot(probs, CLASS_NAMES)

    return render_template('index.html', clas=clas, prob=prob, url=url, plot=plot)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/')
def main():
    return render_template('index.html')
