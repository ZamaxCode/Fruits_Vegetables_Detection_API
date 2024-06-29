import io
import numpy as np
import onnxruntime as ort
from PIL import Image
from ..consts.apiconst import LABELS
from ..utils.image_utils import expand2square, yolobbox2bbox, non_max_suppression_fast

def get_results(image_data):
    preprocess_image = preprocess(image_data)
    detections = execute(preprocess_image)
    postprocess_det = postprocess(detections)
    response = generate_response(postprocess_det)
    return response
    

def preprocess(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = expand2square(image, (128,128,128), (640, 640))
    image = np.array(image).astype(np.float32)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

def execute(image):
    model_path = 'app/weigths/FV_model_v1.onnx'
    session = ort.InferenceSession(model_path)
    outputs = session.run(None, {session.get_inputs()[0].name: image})
    outputs = np.array(outputs).squeeze().transpose()
    detections = []
    for out in outputs:
        sigmoid = max(out[4:])
        label = np.argmax(out[4:])
        if sigmoid >= 0.5:
            detections.append([out[0],out[1],out[2],out[3], label])
    return detections

def postprocess(detections):
    det = np.array(detections)
    x = det[:, 0]
    y = det[:, 1]
    w = det[:, 2]
    h = det[:, 3]
    det[:, :4] = yolobbox2bbox(x,y,w,h)
    new_det = non_max_suppression_fast(det, 0.5)
    return new_det

def generate_response(result):
    labels = result[:,4].tolist()
    occurrence_dict = {}
    for l in labels:
        if LABELS[l] in occurrence_dict:
            occurrence_dict[LABELS[l]] += 1
        else:
            occurrence_dict[LABELS[l]] = 1

    detections=[]
    for label, quantity in occurrence_dict.items():
         detections.append({"name":label,
                            "quantity":quantity})
    response = {
        'detections':detections
    }

    return response