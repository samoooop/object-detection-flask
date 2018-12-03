import base64
from PIL import Image
import cv2
from io import StringIO, BytesIO
import numpy as np

def base64_to_image(base64_string):
    idx = base64_string.find(',')
    if idx != -1:
        base64_string = base64_string[idx + 1:]
    sbuf = StringIO()
    bs = base64.b64decode(base64_string)
    print(bs)
    image = Image.open(BytesIO(bs))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)