"""Compare class and position-preserving patch readouts on controlled glyph arrays."""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from human_alignment.models import VisionModel
from probe_navon_shapes import evaluate

variant = sys.argv[1]
out = Path('outputs/navon_shapes/spatial')
out.mkdir(exist_ok=True)
# Eight anchors per arrangement on a shared grid, with fixed glyph size and ink.
patterns = (
    (1, 3, 5, 7, 9, 15, 17, 19),
    (0, 2, 4, 10, 14, 20, 22, 24),
    (0, 2, 4, 10, 14, 15, 20, 24),
    (2, 5, 9, 10, 14, 15, 19, 22),
    (0, 4, 6, 8, 16, 18, 20, 24),
    (0, 4, 5, 9, 11, 13, 20, 24),
    (0, 2, 4, 8, 16, 20, 22, 24),
)
for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
if not tf.config.list_physical_devices('GPU'):
    raise RuntimeError('Use the Slurm GPU allocation')
model = VisionModel('dino', variant)
height, width = model.input_spec.shape.as_list()[1:3]
font = ImageFont.truetype('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf', 34)
images = []
labels = []
for local, glyph in enumerate('OSAXNZT', 1):
    for outline, anchors in enumerate(patterns, 1):
        image = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(image)
        for anchor in anchors:
            draw.text((width//6 + (anchor%5)*width//6, height//6 + (anchor//5)*height//6),
                      glyph, font=font, anchor='mm', fill='white')
        images.append(np.asarray(image))
        labels.append((outline, local))
x = np.stack(images).astype('float32') / 255
labels = np.array(labels)
np.testing.assert_array_equal(np.broadcast_to(np.stack(images).sum(axis=(1,2,3)).reshape(7,7)[:, :1], (7,7)),
                              np.stack(images).sum(axis=(1,2,3)).reshape(7,7))
readouts = {}
for batch in np.array_split(x, 7):
    prediction = model.predict(batch)
    for layer in (9,10,11):
        tokens = prediction[f'layer_{layer}']
        patch = tokens[:,1:].reshape(-1,37,37,768)
        zones = [block.mean(axis=(1,2)) for row in np.array_split(patch,3,axis=1)
                 for block in np.array_split(row,3,axis=2)]
        for key, features in ((f'class_{layer}', tokens[:,0]), (f'spatial_{layer}', np.concatenate(zones, axis=1)),
                              (f'mean_patch_{layer}', patch.mean(axis=(1,2)))):
            readouts.setdefault(key, []).append(features)
    readouts.setdefault('pre_logits', []).append(prediction['pre_logits'])
result = {}
for key, values in readouts.items():
    z = np.concatenate(values).astype(float)
    z /= np.linalg.norm(z,axis=1,keepdims=True)
    result[key] = evaluate(z, *labels.T)
(out / f'{variant}.json').write_text(json.dumps(result, indent=2))
