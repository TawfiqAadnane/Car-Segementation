import numpy as np
import cv2
from ultralytics import YOLO
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
    

def SegmentCar(objects, image):

    for results in objects:
        boxes = results.boxes
        classe = boxes.cls
        if len(classe)==0:
            print('No car detected')
            break
        classe_names = ["person", "bicycle", "car"]
        output_index = int(classe[0])
        classe_name = classe_names[output_index]

        if len(classe) > 0 and classe[0]==2 :
            xyxy = boxes.xyxy
            x1,y1,x2,y2 = xyxy[0]
        
            #modelURL = "https://huggingface.co/spaces/abhishek/StableSAM/blob/main/sam_vit_h_4b8939.pth"
            sam_checkpoint = './pretrained_models/sam_vit_h_4b8939.pth'
            model_type = "vit_h"

            device = "cpu"

            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)

            predictor = SamPredictor(sam)
            
            predictor.set_image(image)

            input_box = np.array(xyxy[0].tolist())            
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
    #positive_img = positive_img0.astype(np.uint8)*255

    image[positive_img0.all(axis=2)] = [255, 255, 255]

    return image


def applySegment(input_path, outputPath):

    model = YOLO('yolov8n.pt')

    image = cv2.imread(input_path)

    objects = model(image, save=True,classes=[2])

    image_seg = SegmentCar(objects,image)

    cv2.imwrite(outputPath,image_seg)
    return


#example: puth input put path to image 
    
applySegment('00029.jpg', '00029-rem.png')
