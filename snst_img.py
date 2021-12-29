import argparse
#import nst
#import seg_blend as sb
import sys
import time

def snst(img_a_path, img_b_path, style_path, model_path, output_path, epochs, steps_per_epoch, blend_pos):
    segmented_a_path = "tmp/segmented_a.jpg"
    segmented_b_path = "tmp/segmented_b.jpg"

    start = time.time()
    # extract fg from img_a
    sb.segment(img_a_path, segmented_a_path, model_path)
    nst.nst(img_a_path, style_path, epochs, steps_per_epoch)

    # extract bg from img_b
    # perform nst on extracted img_a fg
    nst.opt_performer() 
    #blend
    end = time.time()
    print("Done. Total run time: {:.1f} s / {:.3f} min" .format(end-start, (end-start) / 60))
    

def main():
    version = "1.0"
    p = argparse.ArgumentParser(prog="snst-img.py", description="snst-img: selective neural style transfer for images")
    p.add_argument("-a", "--img-a", type=str, required=True, metavar="", help="foreground image path")
    p.add_argument("-b", "--img-b", type=str, required=True, metavar="", help="background image path")
    p.add_argument("-s", "--style", type=str, required=True,  metavar="", help="style image path")
    p.add_argument("-o", "--output", type=str, metavar="", default="output.png", help="output path (default: ./output.jpg)")
    p.add_argument("-m", "--model", type=str, required=True, metavar="", help="segmentation model path (default: ./model.pkl)")
    p.add_argument("--epochs", type=int, default=10, metavar="", help="number of epochs for NST optimization (default: 10)")
    p.add_argument("--steps-per-epoch", type=int, default=10, metavar="", help="number of steps per epoch for NST optimization (default: 10)")
    p.add_argument("--blend-pos", metavar="", help="blend position of style-transfered img_a in img_b (default: center of img_b)")
    p.add_argument("-v", "--version", action="version", version="%(prog)s "+version,help="print program version and exit")
    args = p.parse_args()

    #snst(args.img_a, args.img_b, args.style, args.output, args.epochs, args.steps_per_epoch, args.blend_pos)
    
if __name__ == "__main__":
    main()
