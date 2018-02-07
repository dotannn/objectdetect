import tarfile
import six.moves.urllib as urllib
import os
from argparse import ArgumentParser


def download_model(model_name=None):
    if model_name is None:
        model_name = 'ssd_mobilenet_v1_coco_2017_11_17'

    MODEL_FILE = model_name + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    opener = urllib.request.URLopener()

    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        # if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd() +"/../models/")


def main():
    parser = ArgumentParser(description="models downloader")
    parser.add_argument("--model", "-m", help="model name", default="ssd_mobilenet_v1_coco_2017_11_17")
    args = parser.parse_args()
    download_model(args.model)


if __name__ == '__main__':
    main()