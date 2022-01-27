import cv2 as cv
import numpy as np
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
from pixellib.tune_bg import alter_bg

def segment(image, model):

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

    segmask, output = ins.segmentImage(image, show_bboxes = True, extract_segmented_objects = True)

    obj = segmask["extracted_objects"][0]

    return (segmask["masks"], obj)


def segment_bg(bg_path, model):

    ins = instanceSegmentation()
    ins.load_model(model)
    segmask, output = ins.segmentImage(bg_path, show_bboxes = True, extract_segmented_objects = True)

    fg_dim = (segmask["extracted_objects"][0].shape[1],
              segmask["extracted_objects"][0].shape[0])

    fg_tr_pos = (segmask["boxes"][0][1], segmask["boxes"][0][0])
    print("object bg pos: {}".format(fg_tr_pos))

    seg_mask = segmask["extracted_objects"][0]
    res = cv.imread(bg_path)

    return (res, segmask["extracted_objects"][0].shape[:2], fg_tr_pos)


def blend(fg_path, bg_path, output_path, blend_pos):

    fg = cv.imread(fg_path, cv.IMREAD_COLOR)
    bg = cv.imread(bg_path, cv.IMREAD_COLOR)

    if (blend_pos > (bg.shape[1], bg.shape[0])):
      raise ValueError("Blending position out of bounds of background.")

    if (fg.size > bg.size):
      raise ValueError("Blend object larger than background.")

    # pārveidot melno par caurspīdīgu
    _fg     = cv.cvtColor(fg, cv.COLOR_BGR2GRAY)
    _, a    = cv.threshold(_fg, 6, 255, cv.THRESH_BINARY)
    b, g, r = cv.split(fg)
    fg      = cv.merge([b, g, r, a], 4)

    fg_a = fg[:, :, 3] / 255.0
    bg_a = 1.0 - fg_a

    print("blend pos: {} / {}" .format(blend_pos, (bg.shape[1], bg.shape[0])))

    h0 = blend_pos[1]
    h1 = h0 + fg.shape[0]
    w0 = blend_pos[0]
    w1 = w0 + fg.shape[1]

    # B(x) = alpha * f(x) + (1-alpha) * g(x), g(x) - bg
    for i in range(0, 3):
      bg[h0:h1, w0:w1, i] = (fg_a * fg[:, :, i]) + bg_a * bg[h0:h1, w0:w1, i]

    cv.imwrite(output_path, bg)


