import argparse

from mmengine.config import Config, DictAction

parser = argparse.ArgumentParser(description='Deepfake Detection Args')
parser.add_argument('--config_file', type=str,
                    default=r'training/config/detector/rgbmsnlc.yaml',
                    help='path to detector YAML file')
parser.add_argument("--opts", action=DictAction, help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
args = parser.parse_args()
cfg = Config.fromfile(args.config_file)
if args.opts is not None:
    cfg.merge_from_dict(args.opts)
pass