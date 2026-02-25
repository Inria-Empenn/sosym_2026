import csv
import importlib
import json
import os
import subprocess

import datalad.api as dl
import numpy as np

from core.data_descriptor import DataDescriptor
from core.file_service import HCP_DS, HCP_EXCLUDED

DATASET_PATH = '/home/ymerel/hcp/'
URL = 'https://github.com/datalad-datasets/human-connectome-project-openaccess.git'

class HcpService:

    def init_dataset(self):
        """
        If needed, initialize the HCP dataset 2 level deep
        :return:
        """
        if os.path.isdir(os.path.join(DATASET_PATH,"HCP1200", "100206", "release-notes")):
            return
        cmd = f"datalad install -R 2 -s {URL} {DATASET_PATH}"
        subprocess.run(cmd.split(" "))

    def read_release_notes(self):
        """
        Extract available images paradigm and type for each subject
        :return:
        """
        dataset = dict()
        path = os.path.join(DATASET_PATH, 'HCP1200')
        for subject in os.listdir(path):
            data = []
            for file in os.listdir(os.path.join(path, subject, 'release-notes')):
                if file.endswith('_unproc.txt'):
                    blocks = file.replace('_unproc.txt', '').split('_')
                    paradigm = blocks[0]
                    ttype = ''
                    if len(blocks) > 1:
                        ttype = blocks[1]
                    if paradigm == 'Structural':
                        ttype = "T2w_SPC1"
                    data.append({'paradigm': paradigm, 'type': ttype})
            dataset[subject] = data
        return dataset

    def write_dataset_desc(self):
        """
        Write dataset description as JSON
        :return:
        """
        dataset = self.read_release_notes()
        with open(HCP_DS, 'w') as json_file:
            json.dump(dataset, json_file, indent=4)

    def download_subject_data(self, subject: str, data_desc: DataDescriptor):
        """
        Get data for a given subject
        :param subject:
        :param data_desc:
        :return:
        """
        paths = [os.path.join(data_desc.data_path, data_desc.func_subpath.replace('{subject_id}', subject)),
                 os.path.join(data_desc.data_path, data_desc.func_subpath.replace('{subject_id}', subject))]

        dataset = self.get_dataset(data_desc.data_path)

        for path in paths:
            dataset.get(path)

    def get_dataset(self, path):
        return dl.Dataset(path)

    def get_excluded_subjects(self):
        with importlib.resources.open_text('hcp', HCP_EXCLUDED) as file:
            reader = csv.reader(file)
            excluded = [row[0] for row in reader]

            if not excluded:
                raise IOError(f"Empty exclusion list from [hcp/{HCP_EXCLUDED}]")

            return excluded

    def sample_subjects(self, size: int):
        """
        Sample subjects from dataset description, excluding those in exclusion list
        :param size:
        :return:
        """
        with importlib.resources.open_text('hcp', HCP_DS) as file:
            dataset = json.load(file)
            filtered = [item for item in list(dataset.keys()) if item not in self.get_excluded_subjects()]
            limit = len(filtered)
            if size > limit:
                raise ValueError(f"Only [{limit}] non-excluded subjects available")
            return np.random.choice(np.array(filtered), size=size, replace=False)
