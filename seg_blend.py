import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
from pixellib.tune_bg import alter_bg
import numpy as np
import cv2 as cv
#!pip install pixellib opencv-python matplotlib numpy torch

def segment(image, output, model):
    # Iespējamās target_classes
    # person, bicycle, car, motorcycle, airplane,
    # bus, train, truck, boat, traffic_light, fire_hydrant, stop_sign,
    # parking_meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra,
    # giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard,
    # sports_ball, kite, baseball_bat, baseball_glove, skateboard, surfboard, tennis_racket,
    # bottle, wine_glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
    # broccoli, carrot, hot_dog, pizza, donut, cake, chair, couch, potted_plant, bed,
    # dining_table, toilet, tv, laptop, mouse, remote, keyboard, cell_phone, microwave,
    # oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy_bear, hair_dryer,
    # toothbrush

    ins = instanceSegmentation()
    # Specifiska segmentācija
    #target_classes = ins.select_target_classes(dog = True)

    ins.load_model(model)
    # Specifiska segmentācija
    #ins.segmentImage(image, show_bboxes=True, segment_target_classes = target_classes, extract_segmented_objects = True, save_extracted_objects = True, output_image_name=output)
    # visi iespējami identificējamie objekti
    ins.segmentImage(image, show_bboxes=True, extract_segmented_objects = True, save_extracted_objects = True, output_image_name=output)

def blend_poisson(fg_path, bg_path, output_path, blend_pos):
    fg = cv.imread(fg_path)
    bg = cv.imread(bg_path)

    if (blend_pos == None):
      blend_pos = (int(bg.shape[1]/2), int(bg.shape[0]/2))

    if (blend_pos > (bg.shape[1], bg.shape[0])):
      raise ValueError("Blending position out of bounds.")

    fg_hsv = cv.cvtColor(fg, cv.COLOR_BGR2HSV)
    mask = cv.inRange(fg_hsv, np.array([1,1,1],dtype=np.uint8), np.array([255,255,255],dtype=np.uint8))
    cont = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cont = cont[0] if len(cont) == 2 else cont[1]
    cv.fillPoly(mask, cont, (255,255,255))

    print("blend pos: {} / {}" .format(blend_pos, (bg.shape[1], bg.shape[0])))
    clone = cv.seamlessClone(fg, bg, mask, blend_pos, cv.MIXED_CLONE)
    cv.imwrite(output_path, clone)
        


