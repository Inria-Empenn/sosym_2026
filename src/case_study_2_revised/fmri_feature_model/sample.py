import math
from argparse import ArgumentParser

from aafm.sample_service import SampleService
from core.file_service import FileService


def sample():
    file_srv = FileService()
    sample_srv = SampleService()


    parser = ArgumentParser(description='Sample configuration')
    parser.add_argument('--nconfig', type=int,  required=True, help='number of configuration to sample')
    parser.add_argument('--parts', type=int, help='number of sample parts', default=1)
    args = parser.parse_args()

    nconfig = int(args.nconfig)
    parts = min(int(args.parts), nconfig)
    configs_per_file = math.ceil(nconfig / parts)

    configs = sample_srv.get_config_sample(nconfig)
    for i in range(parts):
        start = i * configs_per_file
        end = min(start + configs_per_file, nconfig)

        file_srv.write_config2csv(configs[start:end],f"./config_{i+1}.csv")

    file_srv.write_config2csv(sample_srv.get_ref_config(), f"./config_ref.csv")

if __name__ == '__main__':
    sample()
