from pathlib import Path

import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

ALL_CONTOURS = -1

parser = argparse.ArgumentParser()
parser.add_argument("image", help="Input image")
parser.add_argument("masks", nargs="+", help="Masks to overlay")
parser.add_argument("--output", default="output.png", help="Output file")
parser.add_argument("--dpi", default=300, help="DPI of output")
parser.add_argument("--thickness", default=2, help="Thickness of contours")
parser.add_argument(
    "--cmap", default="prism", help="Colour map to use to plot overlays"
)
parser.add_argument(
    "--colours", nargs="+", default=None, help="Hex colours to use for overlays, overrides cmap"
)
parser.add_argument("--labels", nargs="+", help="Overlay labels")


def parse_mask_paths(masks_arg):
    mask_paths = []
    for mask in masks_arg:
        mask_path = Path(mask).resolve()
        assert mask_path.is_file()
        mask_paths.append(mask_path)
    return mask_paths


def parse_labels(labels_arg, masks_arg):
    if labels_arg is not None:
        assert len(labels_arg) == len(masks_arg)
    return labels_arg


def parse_file_path(path_arg):
    path = Path(path_arg).resolve()
    assert path.is_file()
    return path


def main():
    args = parser.parse_args()

    image_path = parse_file_path(args.image)

    labels = parse_labels(args.labels, args.masks)

    mask_paths = parse_mask_paths(args.masks)

    output_path = Path(args.output).resolve()

    dpi = int(args.dpi)

    thickness = int(args.thickness)

    if args.colours is None:
        cmap = plt.get_cmap(args.cmap)
        overlay_colours = cmap(np.linspace(0.1, 0.9, len(mask_paths)))
    else:
        overlay_colours = np.array([mcolors.to_rgba(h) for h in args.colours])

    # Plot input image. Note that we first need to reverse the order of RGB storage before plotting with matplotlib.
    image = cv.imread(str(image_path))
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    print(overlay_colours)

    patches = []
    for i, mask_path in enumerate(mask_paths):
        # Open and binarise mask.
        mask = cv.imread(str(mask_path))
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

        # We create an empty RGBA image to plot the contours onto.
        height, width, _ = image.shape
        contours_img = np.zeros((height, width, 4), dtype=np.uint8)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # OpenCV expects values between 0 and 255.
        contour_color = overlay_colours[i] * 255
        cv.drawContours(contours_img, contours, ALL_CONTOURS, contour_color, thickness)

        plt.imshow(contours_img)
        del contours_img

        if labels is not None:
            patches.append(mpatches.Patch(color=overlay_colours[i], label=labels[i]))

    if labels is not None:
        plt.legend(handles=patches)

    plt.axis("off")
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    main()
