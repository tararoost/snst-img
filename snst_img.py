import argparse
import cv2 as cv
import numpy as np
import os
import time

import seg_blend
import fast_nst

def snst(img_path, style_path, model_path, model_fast_nst, output_path):
    ld = os.path.dirname(__file__)
    md = os.path.join(ld, "tmp")
    os.makedirs(md, exist_ok = True)

    # temporary working files
    segmented_obj_path = "tmp/segmented_obj.jpg"
    style_transfered_path = "tmp/style_transfered.jpg"

    start = time.time()

    segmented_obj = seg_blend.segment(img_path, model_path)
    cv.imwrite(segmented_obj_path, segmented_obj[1])

    style_transfered_obj = fast_nst.fast_nst(segmented_obj_path, style_path, model_fast_nst, 256)

    _st = np.array(style_transfered_obj)
    _st = cv.resize(_st, (segmented_obj[1].shape[1], segmented_obj[1].shape[0]), interpolation = cv.INTER_AREA)
    seg_arr = segmented_obj[1][:,:,:3]
    mask = cv.cvtColor((_st * seg_arr), cv.COLOR_BGRA2GRAY)
    _, mask = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)
    res = cv.bitwise_and(_st, _st, mask=mask)
    cv.imwrite(style_transfered_path, res[:,:,::-1])

    segmented_bg = seg_blend.segment_bg(img_path, model_path)

    seg_blend.blend(style_transfered_path, img_path, output_path,
                    (segmented_bg[2][1], segmented_bg[2][0]))

    end = time.time()
    print("Done. Total run time: {:.1f} s" .format(end-start))
    

def main():
    version = "1.0"
    p = argparse.ArgumentParser(prog="snst-img.py", description="snst-img: selective neural style transfer for images")

    p.add_argument("-i", "--img", type=str, required=True, metavar="",
                   help="image path")

    p.add_argument("-s", "--style", type=str, required=True,  metavar="",
                   help="style image path")

    p.add_argument("-o", "--output", type=str, metavar="", default="output.png",
                   help="output path (default: ./output.png)")

    p.add_argument("-m", "--model", type=str, required=True, metavar="", default="model.pkl",
                   help="segmentation model path (default: ./model.pkl)")

    p.add_argument("-m_fnst", "--model-fast-nst", type=str, required=False, metavar="",
                   help="arbitrary image stylization tensorflow hub module archive path (if none, module will be downloaded)")

    p.add_argument("-v", "--version", action="version", version="%(prog)s "+version,
                   help="print program version and exit")

    args = p.parse_args()

    snst(args.img, args.style, args.model, args.model_fast_nst, args.output)
    
if __name__ == "__main__":
    main()
