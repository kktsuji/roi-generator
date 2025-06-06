"""Generate a region of interest (ROI) for a given image."""

import cv2
from dotenv import load_dotenv
import numpy as np

import glob
import os
import csv


def _extract_and_save_roi(base_root_dir_path, base_out_dir_path):
    CSV_ROOT_PATH = base_root_dir_path + "source_code/CADforCTCs/CADforCTCs/output/"
    CSV_ROOT_PATH_LIST = glob.glob(CSV_ROOT_PATH + "*tif/")
    IMG_ROOT_PATH = base_root_dir_path + "data/"
    ROI_SIZE = 40
    ROI_HALF_SIZE = ROI_SIZE // 2

    for csv_root_path in CSV_ROOT_PATH_LIST:
        dir_name = os.path.basename(csv_root_path[:-1])
        print(csv_root_path)
        print(dir_name)

        out_dir_path = base_out_dir_path + dir_name + "/"
        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)

        # csv: image no., x, y
        csv_path = csv_root_path + "CSVs/groundTruthCTCs.csv"
        with open(csv_path, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            data = list(reader)
            data = [[int(value) for value in row] for row in data]

        img_dir_path = glob.glob(IMG_ROOT_PATH + dir_name + "/*after*")[0] + "/"

        for row in data:
            img_no, x, y = row
            img_path_glob = img_dir_path + "*s" + str(img_no).zfill(3) + "*.tif"
            img_path_list = glob.glob(img_path_glob)
            print("  -", img_path_glob)

            # Load 16-bit TIFF images as original pixel values
            img_b = cv2.imread(img_path_list[1], cv2.IMREAD_UNCHANGED)  # ch01
            img_g = cv2.imread(img_path_list[2], cv2.IMREAD_UNCHANGED)  # ch02
            img_r = cv2.imread(img_path_list[3], cv2.IMREAD_UNCHANGED)  # ch03
            if img_b is None or img_g is None or img_r is None:
                print("  - Error: Image not found or could not be read.")
                exit()

            height, width = img_b.shape
            y1 = max(y - ROI_HALF_SIZE, 0)
            y2 = min(y + ROI_HALF_SIZE, height)
            x1 = max(x - ROI_HALF_SIZE, 0)
            x2 = min(x + ROI_HALF_SIZE, width)

            # adjust ROI size
            if y2 - y1 < ROI_SIZE:
                if y1 == 0:
                    y2 = min(y2 + (ROI_SIZE - (y2 - y1)), height)
                else:
                    y1 = max(y1 - (ROI_SIZE - (y2 - y1)), 0)
            if x2 - x1 < ROI_SIZE:
                if x1 == 0:
                    x2 = min(x2 + (ROI_SIZE - (x2 - x1)), width)
                else:
                    x1 = max(x1 - (ROI_SIZE - (x2 - x1)), 0)

            roi_b = img_b[y1:y2, x1:x2]
            roi_g = img_g[y1:y2, x1:x2]
            roi_r = img_r[y1:y2, x1:x2]

            if (
                roi_b.shape != (ROI_SIZE, ROI_SIZE)
                or roi_g.shape != (ROI_SIZE, ROI_SIZE)
                or roi_r.shape != (ROI_SIZE, ROI_SIZE)
            ):
                print(
                    f"  - Error: ROI size mismatch for img_no={img_no}, x={x}, y={y}. "
                    f"Expected ({ROI_SIZE}, {ROI_SIZE}), got "
                    f"{roi_b.shape}, {roi_g.shape}, {roi_r.shape}."
                )
                exit()

            # Save as grayscale 16-bit png images without compression
            cv2.imwrite(
                out_dir_path + f"img-no{str(img_no).zfill(3)}_y{y}_x{x}_b.png",
                roi_b.astype(np.uint16),
            )
            cv2.imwrite(
                out_dir_path + f"img-no{str(img_no).zfill(3)}_y{y}_x{x}_g.png",
                roi_g.astype(np.uint16),
            )
            cv2.imwrite(
                out_dir_path + f"img-no{str(img_no).zfill(3)}_y{y}_x{x}_r.png",
                roi_r.astype(np.uint16),
            )


def _round(src, decimals=0):
    return np.floor(src * 10**decimals + 0.5) / 10**decimals


def _make_pseudo_rgb_image(base_out_dir_path):
    BIT_DEPTH_IN = 16
    BIT_DEPTH_OUT = 8
    MAX_VALUE_DIGIT_IN = int(pow(2, BIT_DEPTH_IN) - 1)
    MAX_VALUE_DIGIT_OUT = int(pow(2, BIT_DEPTH_OUT) - 1)

    OUT_DIR_PATH = base_out_dir_path + "pseudo_rgb/"
    if not os.path.exists(OUT_DIR_PATH):
        os.makedirs(OUT_DIR_PATH)

    roi_dir_list = glob.glob(base_out_dir_path + "*/")
    for roi_dir in roi_dir_list:
        if roi_dir == OUT_DIR_PATH:
            continue

        dir_name = os.path.basename(roi_dir[:-1]).replace(" ", "-")
        print(roi_dir)
        print("  -", dir_name)

        roi_path_list = glob.glob(roi_dir + "*_b.png")

        for roi_path in roi_path_list:
            roi_name = (
                dir_name + "_" + os.path.basename(roi_path).replace("_b.png", ".png")
            )
            print("    -", roi_name)

            roi_b = cv2.imread(roi_path, cv2.IMREAD_UNCHANGED)
            roi_g = cv2.imread(
                roi_path.replace("_b.png", "_g.png"), cv2.IMREAD_UNCHANGED
            )
            roi_r = cv2.imread(
                roi_path.replace("_b.png", "_r.png"), cv2.IMREAD_UNCHANGED
            )
            if roi_b is None or roi_g is None or roi_r is None:
                print("  - Error: ROI image not found or could not be read.")
                exit()

            pseudo_rgb_image = cv2.merge((roi_b, roi_g, roi_r))

            pseudo_rgb_image_normalized = (
                pseudo_rgb_image.astype(np.float64)
                * MAX_VALUE_DIGIT_OUT
                / MAX_VALUE_DIGIT_IN
            )
            pseudo_rgb_image_normalized = _round(pseudo_rgb_image_normalized)
            pseudo_rgb_image_normalized = np.clip(
                pseudo_rgb_image_normalized, 0, MAX_VALUE_DIGIT_OUT
            )
            pseudo_rgb_image_normalized = pseudo_rgb_image_normalized.astype(np.uint8)

            cv2.imwrite(
                OUT_DIR_PATH + roi_name,
                pseudo_rgb_image_normalized,
                [cv2.IMWRITE_PNG_COMPRESSION, 0],  # No compression
            )


if __name__ == "__main__":
    load_dotenv()
    ROOT_DIR_PATH = os.environ.get("ROOT_DIR_PATH")
    OUT_DIR_PATH = os.environ.get("OUT_DIR_PATH")

    _extract_and_save_roi(ROOT_DIR_PATH, OUT_DIR_PATH)
    _make_pseudo_rgb_image(OUT_DIR_PATH)
