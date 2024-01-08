#!/usr/bin/env python3

import wget
import zipfile
import os

DATA_DIR = os.path.dirname(__file__)
KITTI_DIR = "kitti"

if not os.path.exists(KITTI_DIR):
    os.mkdir(KITTI_DIR)

os.chdir(KITTI_DIR)

datasets = [
    {'name': '2011_09_26', 'indices': ['0106']}
]


def download(url):
    print(f"downloading: {url}\n")
    zip = wget.download(url)
    with zipfile.ZipFile(zip, 'r') as zip_ref:
        zip_ref.extractall(".")


for dataset in datasets:
    url_prefix = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"
    download(f"{url_prefix}{dataset['name']}_calib.zip")
    for index in dataset['indices']:
        download(f"{url_prefix}{dataset['name']}_drive_{index}/{dataset['name']}_drive_{index}_sync.zip")
