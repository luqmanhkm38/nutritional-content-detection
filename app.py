from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


app = Flask(__name__)

utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile
PATH_TO_LABELS= 'model/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
detection_model = tf.saved_model.load("model/saved_model")

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    
    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

def show_inference(model, image_path):
    image_np = np.array(Image.open(image_path))
    output_dict = run_inference_for_single_image(model, image_np)

    terdeteksi = []

    for i in range(len(output_dict['detection_classes'])):
        if output_dict['detection_scores'][i] >= 0.5:
            if category_index[output_dict['detection_classes'][i]]['name'] not in terdeteksi:
                terdeteksi.append(category_index[output_dict['detection_classes'][i]]['name'])
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    img = Image.fromarray(image_np)
    return terdeteksi, img

@app.route("/", methods=['GET'])
def open_app():
    return render_template("index.html")

@app.route("/", methods=['POST'])
def run_app():
    imagefile=request.files['file_gambar']
    image_path="./static/images/" + str(imagefile.filename)
    imagefile.save(image_path)
    hasil=show_inference(detection_model, image_path)
    hasil[1].save(image_path)
    data_gizi={'Kentang' : 'Informasi Nilai Kandungan Per 100 g : <br> Karbohidrat : 17.72 g <br> Protein : 1.37 g <br> Natrium : 27 mg <br> Gula 5.74 g <br> Vitamin A : 787 mcg <br> Vitamin C : 12.80 mg', 'Wortel' : 'Informasi Nilai Kandungan Per 100 g : <br> Karbohidrat : 7.90 g <br> Protein : 1 g <br> Natrium : 70 mg <br> Kalsium 45 mg <br> Vitamin B2 : 0.04 mg <br> Vitamin C : 18 mg', 'Jeruk' : 'Informasi Nilai Kandungan Per 100 g : <br> Karbohidrat : 11.20 g <br> Protein : 0.90 g <br> Natrium : 4 mg <br> Kalsium 33 mg <br> Vitamin B2 : 0.03 mg <br> Vitamin C : 49 mg', 'Pisang' : 'Informasi Nilai Kandungan Per 100 g : <br> Karbohidrat : 26.30 g <br> Protein : 0.80 g <br> Natrium : 10 mg <br> Kalsium 10 mg <br> Vitamin B1 : 0.10 mcg <br> Vitamin C : 9 mg', 'Apel' : 'Informasi Nilai Kandungan Per 100 g : <br> Karbohidrat : 14.90 g <br> Protein : 0.30 g <br> Natrium : 2 mg <br> Kalsium 6 mg <br> Vitamin B3 : 0.10 mg <br> Vitamin C : 5 mg', 'Brokoli' : 'Informasi Nilai Kandungan Per 100 g : <br> Karbohidrat : 1.90 g <br> Protein : 3.20 g <br> Natrium : 15 mg <br> Kalsium 298 mg <br> Vitamin A : 137 mcg <br> Vitamin C : 61.10 mg'}
    eng_indo={'Banana' : 'Pisang', 'Apple' : 'Apel', 'Orange' : 'Jeruk', 'Broccoli' : 'Brokoli', 'Potato' : 'Kentang', 'Carrot' : 'Wortel'}

    gizi_result=[]
    buah_indo=[]
    for i in hasil[0]:
        buah_indo.append(eng_indo[i])
    for j in buah_indo:
        gizi_result.append(data_gizi[j])
    
    path_image="./images/" + str(imagefile.filename)


    return render_template("result.html", hasil_gambar=str(path_image), hasil_deteksi=buah_indo, gizi_deteksi=gizi_result)

if __name__=="__main__":
    app.run(debug=True)