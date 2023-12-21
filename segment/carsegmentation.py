import numpy as np
import cv2
from ultralytics import YOLO
import easygui
import sys
from segment_anything import sam_model_registry, SamPredictor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   


model = YOLO('yolov8n.pt')

image = easygui.fileopenbox(title='Select input file', filetypes=['*.jpg', '*.png'])

objects = model(image, save=True,classes=[2])


def SegmentCar(objects):
    for results in objects:
        boxes = results.boxes
        classe = boxes.cls
        if len(classe)==0:
            print('No car detected')
            break
        classe_names = ["person", "bicycle", "car"]
        output_index = int(classe[0])
        print(output_index)
        classe_name = classe_names[output_index]
        #print(classe_names)
        if len(classe) > 0 and classe[0]==2 :
            xyxy = boxes.xyxy
            x1,y1,x2,y2 = xyxy[0]

            sam_checkpoint = "/kaggle/input/segment-anything-models/sam_vit_h_4b8939.pth"
            model_type = "vit_h"

            device = "cuda"

            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)

            predictor = SamPredictor(sam)
            
            predictor.set_image(image)
            input_box = np.array(xyxy[0].tolist())
            print("input Box", input_box)
            
            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None,:],
                multimask_output=False,
            )
    mask = masks[0]
    
    negative_img0 = np.tile(mask[:,:,np.newaxis],(1,1,3)).astype(int)
    negative_img = negative_img0*255
    positive_img0 = np.logical_not(negative_img)
    positive_img0 = positive_img0.astype(int)
    positive_img = positive_img0.astype(np.uint8)*255

    image[positive_img0.all(axis=2)] = [255, 255, 255]

    return image