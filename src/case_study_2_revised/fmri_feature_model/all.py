import math
from argparse import ArgumentParser

from aafm.sample_service import SampleService
from core.file_service import FileService


def sample():
    file_srv = FileService()
    sample_srv = SampleService()

    file_srv.write_config2csv(sample_srv.get_all_configs(), f"./config_all.csv")

if __name__ == '__main__':
    sample()
