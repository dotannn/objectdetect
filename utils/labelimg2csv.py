import os
import glob
from argparse import ArgumentParser

import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    parser = ArgumentParser(description="convert labelimage xmls to one csv", usage="python labelimg2csv.py --path <annotated_images_dir> --output <labels.csv file>")

    parser.add_argument("--path", "-p", help="annotated images path", default="../data/annotations/")
    parser.add_argument("--output", "-o", help="output file", default="../data/labels.csv")

    args = parser.parse_args()
    xml_df = xml_to_csv(args.path)
    xml_df.to_csv(args.output, index=None)
    print('Successfully converted xml to csv.')


if __name__ == '__main__':
    main()