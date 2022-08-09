"""Save model weights files."""
import argparse
import logging
import os
import sys
import model_cfg

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Save model weights files")
    parser.add_argument("-m", "--model-name", action='append', choices=model_cfg.get_model_names(),
                        help="Model name (default: all models)")
    args = parser.parse_args()

    model_names = model_cfg.get_model_names() if args.model_name is None else args.model_name
    for model_name in model_names:
        model_file = model_cfg.get_model_default_weights_file(model_name)
        if os.path.exists(model_file):
            logger.info('%s: weights file already exists: %s', model_name, model_file)
        else:
            logger.info('%s: saving weights file: %s', model_name, model_file)
            model_cfg.save_model_weights_file(model_name, model_file=model_file)
