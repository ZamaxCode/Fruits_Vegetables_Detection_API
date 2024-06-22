import io
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from ..consts.apiconst import LABELS

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
    response = {
        'detections':occurrence_dict
    }

    return response

def yolobbox2bbox(x,y,w,h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    xyxy = np.array([x1, y1, x2, y2]).transpose()
    return xyxy 


def expand2square(pil_img, background_color, resize):
    width, height = pil_img.size
    if width == height:
        result  = pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
    result = result.resize(resize)
    return result

def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	pick = []

	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		overlap = (w * h) / area[idxs[:last]]

		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	return boxes[pick].astype("int")