# pip install -q "paddlepaddle==2.5.0rc0"
# pip install -q "pyclipper>=1.2.1" "shapely>=1.7.1"

import os
import sys
import cv2
import numpy as np
import paddle
import math
import time
import collections
from PIL import Image
import tarfile
import requests

import re
from flask import Flask, render_template, Response
import threading

from openvino.runtime import Core
from IPython import display
import copy

from utils import viedo_utils
import pre_post_processing as processing


app = Flask(__name__)




det_model_file_path = ".\model\ch_PP-OCRv3_det_infer\inference.pdmodel"


#### Load the Model for Text **Detection**
# Initialize OpenVINO Runtime for text detection.
core = Core()
det_model = core.read_model(model=det_model_file_path)
det_compiled_model = core.compile_model(model=det_model, device_name="AUTO")

# Get input and output nodes for text detection.
det_input_layer = det_compiled_model.input(0)
det_output_layer = det_compiled_model.output(0)






#### Download the Model for Text **Recognition**
#rec_model_url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar"
#rec_model_file_path = Path("model/ch_PP-OCRv3_rec_infer/inference.pdmodel")
#
#run_model_download(rec_model_url, rec_model_file_path)


rec_model_file_path = ".\model\ch_PP-OCRv3_rec_infer\inference.pdmodel"



#### Load the Model for Text **Recognition** with Dynamic Shape
"""
Input to text recognition model refers to detected bounding boxes with different image sizes, for example, dynamic input shapes. Hence:

1. Input dimension with dynamic input shapes needs to be specified before loading text recognition model.
2. Dynamic shape is specified by assigning -1 to the input dimension or by setting the upper bound of the input dimension using, for example, `Dimension(1, 512)`.
"""
# Read the model and corresponding weights from a file.
rec_model = core.read_model(model=rec_model_file_path)

# Assign dynamic shapes to every input layer on the last dimension.
for input_layer in rec_model.inputs:
    input_shape = input_layer.partial_shape
    input_shape[3] = -1
    rec_model.reshape({input_layer: input_shape})

rec_compiled_model = core.compile_model(model=rec_model, device_name="AUTO")

# Get input and output nodes.
rec_input_layer = rec_compiled_model.input(0)
rec_output_layer = rec_compiled_model.output(0)



### Preprocessing Image Functions for Text Detection and Recognition
"""
Define preprosessing functions for text detection and recognition:
1. Preprocessing for text detection: resize and normalize input images.
2. Preprocessing for text recognition: resize and normalize detected box images to the same size (for example, `(3, 32, 320)` size for images with Chinese text) for easy batching in inference.
"""

# Preprocess for text detection.
def image_preprocess(input_image, size):
    """
    Preprocess input image for text detection

    Parameters:
        input_image: input image 
        size: value for the image to be resized for text detection model
    """
    img = cv2.resize(input_image, (size, size))
    img = np.transpose(img, [2, 0, 1]) / 255
    img = np.expand_dims(img, 0)
    # NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
    img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std
    return img.astype(np.float32)
    
    
# Preprocess for text recognition.
def resize_norm_img(img, max_wh_ratio):
    """
    Resize input image for text recognition

    Parameters:
        img: bounding box image from text detection 
        max_wh_ratio: value for the resizing for text recognition model
    """
    rec_image_shape = [3, 48, 320]
    imgC, imgH, imgW = rec_image_shape
    assert imgC == img.shape[2]
    character_type = "ch"
    if character_type == "ch":
        imgW = int((32 * max_wh_ratio))
    h, w = img.shape[:2]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


def prep_for_rec(dt_boxes, frame):
    """
    Preprocessing of the detected bounding boxes for text recognition

    Parameters:
        dt_boxes: detected bounding boxes from text detection 
        frame: original input frame 
    """
    ori_im = frame.copy()
    img_crop_list = [] 
    for bno in range(len(dt_boxes)):
        tmp_box = copy.deepcopy(dt_boxes[bno])
        img_crop = processing.get_rotate_crop_image(ori_im, tmp_box)
        img_crop_list.append(img_crop)
        
    img_num = len(img_crop_list)
    # Calculate the aspect ratio of all text bars.
    width_list = []
    for img in img_crop_list:
        width_list.append(img.shape[1] / float(img.shape[0]))
    
    # Sorting can speed up the recognition process.
    indices = np.argsort(np.array(width_list))
    return img_crop_list, img_num, indices


def batch_text_box(img_crop_list, img_num, indices, beg_img_no, batch_num):
    """
    Batch for text recognition

    Parameters:
        img_crop_list: processed detected bounding box images 
        img_num: number of bounding boxes from text detection
        indices: sorting for bounding boxes to speed up text recognition
        beg_img_no: the beginning number of bounding boxes for each batch of text recognition inference
        batch_num: number of images for each batch
    """
    norm_img_batch = []
    max_wh_ratio = 0
    end_img_no = min(img_num, beg_img_no + batch_num)
    for ino in range(beg_img_no, end_img_no):
        h, w = img_crop_list[indices[ino]].shape[0:2]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)
    for ino in range(beg_img_no, end_img_no):
        norm_img = resize_norm_img(img_crop_list[indices[ino]], max_wh_ratio)
        norm_img = norm_img[np.newaxis, :]
        norm_img_batch.append(norm_img)

    norm_img_batch = np.concatenate(norm_img_batch)
    norm_img_batch = norm_img_batch.copy()
    return norm_img_batch    
    
    
    
    
    
    
### Postprocessing Image for Text Detection
def post_processing_detection(frame, det_results):
    """
    Postprocess the results from text detection into bounding boxes

    Parameters:
        frame: input image 
        det_results: inference results from text detection model
    """   
    ori_im = frame.copy()
    data = {'image': frame}
    data_resize = processing.DetResizeForTest(data)
    data_list = []
    keep_keys = ['image', 'shape']
    for key in keep_keys:
        data_list.append(data_resize[key])
    img, shape_list = data_list

    shape_list = np.expand_dims(shape_list, axis=0) 
    pred = det_results[0]    
    if isinstance(pred, paddle.Tensor):
        pred = pred.numpy()
    segmentation = pred > 0.3

    boxes_batch = []
    for batch_index in range(pred.shape[0]):
        src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
        mask = segmentation[batch_index]
        boxes, scores = processing.boxes_from_bitmap(pred[batch_index], mask, src_w, src_h)
        boxes_batch.append({'points': boxes})
    post_result = boxes_batch
    dt_boxes = post_result[0]['points']
    dt_boxes = processing.filter_tag_det_res(dt_boxes, ori_im.shape)    
    return dt_boxes    
    




    
### Main Processing Function for PaddleOCR
"""
Run `paddleOCR` function in different operations, either a webcam or a video file. See the list of procedures below:

1. Create a video player to play with target fps (`utils.VideoPlayer`).
2. Prepare a set of frames for text detection and recognition.
3. Run AI inference for both text detection and recognition.
4. Visualize the results.
"""
def run_paddle_ocr(source=0, flip=False, use_popup=True, skip_first_frames=0):
    """
    Main function to run the paddleOCR inference:
    1. Create a video player to play with target fps (utils.VideoPlayer).
    2. Prepare a set of frames for text detection and recognition.
    3. Run AI inference for both text detection and recognition.
    4. Visualize the results.

    Parameters:
        source: The webcam number to feed the video stream with primary webcam set to "0", or the video path.  
        flip: To be used by VideoPlayer function for flipping capture image.
        use_popup: False for showing encoded frames over this notebook, True for creating a popup window.
        skip_first_frames: Number of frames to skip at the beginning of the video. 
    """
    # Create a video player to play with target fps.
    player = None
    try:
        player = viedo_utils.VideoPlayer(source=source, flip=flip, fps=30, skip_first_frames=skip_first_frames)
        # Start video capturing.
        player.start()
        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        reslist = []
        max_num = 0
        
        processing_times = collections.deque()
        while True:
            # Grab the frame.
            frame = player.next()
            if frame is None:
                print("Source ended")
                break
            # If the frame is larger than full HD, reduce size to improve the performance.
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(src=frame, dsize=None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
            # Preprocess the image for text detection.
            test_image = image_preprocess(frame, 640)
                
            # Measure processing time for text detection.
            start_time = time.time()
            # Perform the inference step.
            det_results = det_compiled_model([test_image])[det_output_layer]
            stop_time = time.time()

            # Postprocessing for Paddle Detection.
            dt_boxes = post_processing_detection(frame, det_results)

            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()
            processing_time_det = np.mean(processing_times) * 1000

            # Preprocess detection results for recognition.
            dt_boxes = processing.sorted_boxes(dt_boxes)  
            batch_num = 6
            img_crop_list, img_num, indices = prep_for_rec(dt_boxes, frame)
            
            # For storing recognition results, include two parts:
            # txts are the recognized text results, scores are the recognition confidence level. 
            rec_res = [['', 0.0]] * img_num
            txts = [] 
            scores = []

            for beg_img_no in range(0, img_num, batch_num):

                # Recognition starts from here.
                norm_img_batch = batch_text_box(
                    img_crop_list, img_num, indices, beg_img_no, batch_num)

                # Run inference for text recognition. 
                rec_results = rec_compiled_model([norm_img_batch])[rec_output_layer]

                # Postprocessing recognition results.
                postprocess_op = processing.build_post_process(processing.postprocess_params)
                rec_result = postprocess_op(rec_results)
                for rno in range(len(rec_result)):
                    rec_res[indices[beg_img_no + rno]] = rec_result[rno]   
                if rec_res:
                    txts = [rec_res[i][0] for i in range(len(rec_res))] 
                    scores = [rec_res[i][1] for i in range(len(rec_res))]
                    
                    
                                   
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            # Draw text recognition results beside the image.
            draw_img = processing.draw_ocr_box_txt(
                image,
                boxes,
                txts,
                scores,
                drop_score=0.5)

            # Visualize the PaddleOCR results.
            f_height, f_width = draw_img.shape[:2]
            fps = 1000 / processing_time_det
            cv2.putText(img=draw_img, text=f"Inference time: {processing_time_det:.1f}ms ({fps:.1f} FPS)", 
                        org=(20, 40),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=f_width / 1000,
                        color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
            
            
            # result text 추출
            if len(txts) == 2 and scores[1] > 0.8 :
                print(f"run_paddle_ocr result txts(scores) : [ {txts[1]} ] => [ {scores[1]} ]")
                
                numbers = re.sub(r'[^0-9]', '', txts[1])
                if len(numbers) > 3:
                    numbers = numbers[-3:]
                reslist.append(numbers[-3:])
                
                # result text 
                if len(reslist) > 5:
                    global resData
                    max_num = max(set(reslist), key=reslist.count)
                    resData = max_num 
                    print(f"detection result => {max_num}, {resData}")
                    reslist = []
                # End if
                
            # End if
            
            

            
            
            # Use this workaround if there is flickering.
            if use_popup: 
                draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
                cv2.imshow(winname=title, mat=draw_img)
                key = cv2.waitKey(1)
                # escape = 27
                if key == 27:
                    break
            else:
                # Encode numpy array to jpg.
                draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
                _, encoded_img = cv2.imencode(ext=".jpg", img=draw_img,
                                              params=[cv2.IMWRITE_JPEG_QUALITY, 100])
                # Create an IPython image.
                i = display.Image(data=encoded_img)
                # Display the image in this notebook.
                display.clear_output(wait=True)
                display.display(i)
        
        # End while
        
    # End try    

        
    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            # Stop capturing.
            player.stop()
        if use_popup:
            cv2.destroyAllWindows()




## Run Live PaddleOCR with OpenVINO
"""
Use a webcam as the video input. By default, the primary webcam is set with `source=0`. If you have multiple webcams, each one will be assigned a consecutive number starting at 0. Set `flip=True` when using a front-facing camera. Some web browsers, especially Mozilla Firefox, may cause flickering. If you experience flickering, set `use_popup=True`. 

> **NOTE**: Popup mode may not work if you run this notebook on a remote computer.

Run live PaddleOCR:
"""








@app.route('/')
def index():
    return "000"

@app.route('/num')
def num():
    global resData
    return str(resData)


ip_address = '10.10.14.3'
port = 8080

if __name__ == '__main__':
    #global resData
    resData = -1
    
    t = threading.Thread(target=run_paddle_ocr)
    t.start()
    #run_paddle_ocr(source=0, flip=False, use_popup=True)

    app.run(host=ip_address, port=port, debug=False)  # debug True 하면 전역변수가 동작하지 않는다.
    
# End if

# video file
# Test OCR results on a video file.
#video_file = "https://raw.githubusercontent.com/yoyowz/classification/master/images/test.mp4"
#run_paddle_ocr(source=video_file, flip=False, use_popup=False, skip_first_frames=0)