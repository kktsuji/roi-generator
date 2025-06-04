"""Generate a region of interest (ROI) for a given image."""

import cv2
from dotenv import load_dotenv
import numpy as np

import glob
import os
import csv


if __name__ == "__main__":
    load_dotenv()
    ROOT_DIR_PATH = os.environ.get("ROOT_DIR_PATH")
    CSV_ROOT_PATH = ROOT_DIR_PATH + "source_code/CADforCTCs/CADforCTCs/output/"
    CSV_ROOT_PATH_LIST = glob.glob(CSV_ROOT_PATH + "*tif/")

    for csv_root_path in CSV_ROOT_PATH_LIST:
        dir_name = os.path.basename(csv_root_path[:-1])
        print(csv_root_path)
        print(dir_name)

        # csv: image no., x, y
        csv_path = csv_root_path + "CSVs/groundTruthCTCs.csv"
        with open(csv_path, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            data = list(reader)
            data = [[int(value) for value in row] for row in data]

        print(data)

        exit()
