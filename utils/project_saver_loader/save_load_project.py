import os
import sys
import zipfile

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(cur_path))
sys.path.append(root_path)


def save(output_filename):
    mode = 'w'
    source_dirs = [os.path.join(root_path, 'main_net_classes'),
                   os.path.join(root_path, 'ontologies'),
                   os.path.join(root_path, 'trained_models')]
    for source_dir in source_dirs:
        relroot = os.path.abspath(os.path.join(source_dir, os.pardir))
        with zipfile.ZipFile(output_filename, mode, zipfile.ZIP_DEFLATED) as zip:
            for root, dirs, files in os.walk(source_dir):
                if '__pycache__' in root:
                    continue
                zip.write(root, os.path.relpath(root, relroot))
                for file in files:
                    filename = os.path.join(root, file)
                    if os.path.isfile(filename):
                        arcname = os.path.join(os.path.relpath(root, relroot), file)
                        zip.write(filename, arcname)
            mode = 'a'


def load(filename):
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(root_path)
