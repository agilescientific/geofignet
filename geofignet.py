# -*- coding: utf-8 -*-
"""
Make inferences from geofignet.
"""
from io import BytesIO
import base64

import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


DATA_TRANSFORMS = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def infer(model, image):
    """
    Given a model and an image, make a prediction.

    Returns class, probability.
    """
    device = torch.device("cpu")
    image = DATA_TRANSFORMS(image.convert('RGB')).unsqueeze(0).to(device)
    sm = torch.nn.Softmax(dim=1)
    probs = sm(model(image))
    return probs.detach().numpy().squeeze().tolist()


def plot(probs, class_names):
    """
    Make a plot and return a base64 encoded string.
    """
    y = list(range(len(probs)))
    y_min, y_max = y[0]-0.75, y[-1]+0.75

    fig, ax = plt.subplots(figsize=(6, 10))
    bars = ax.barh(y, probs, color='orange', align='center', edgecolor='none')
    bars[np.argmax(probs)].set_color('red')
    ax.set_yticks(y)
    ax.set_yticklabels(class_names, size=12)
    ax.set_xscale('log')
    ax.set_ylim(y_max, y_min)  # Label top-down.
    ax.grid(c='black', alpha=0.15, which='both')
    ax.patch.set_facecolor("white")
    fig.patch.set_facecolor("none")

    for i, p in enumerate(probs):
        ax.text(0.5*min(probs), i, "{:0.2e}".format(p), va='center')

    plt.tight_layout()

    handle = BytesIO()
    plt.savefig(handle, format='png', facecolor=fig.get_facecolor())
    handle.seek(0)
    figdata_png = base64.b64encode(handle.getvalue())

    return figdata_png.decode('utf8')
