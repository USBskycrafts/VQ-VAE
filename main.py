import argparse
import os

import yaml
from pytorch_lightning import Trainer

from autoencoder.dataset import Dataset
from autoencoder.kernel.model import VQVAE
from autoencoder.utils import load_instance


def build_argparser():
    parser = argparse.ArgumentParser(description="A script to process data.")
    parser.add_argument('--gpus', type=str, default='0', help='GPU ids to use')
    parser.add_argument('--config', type=str,
                        default='config.yaml', help='Path to the config file')
    parser.add_argument('--resume', type=str,
                        default=None, help='Path to the checkpoint to resume from')
    parser.add_argument('--test-only', type=bool, default=False,
                        help='Whether to only test the model')
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    # Set the GPU ids
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # Load the config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create the model
    model = load_instance(config['model'])
    dataset = Dataset(config['dataset'])

    if args.resume:
        model = VQVAE.load_from_checkpoint(args.resume, map_location='cpu')
        print(f"Model loaded from checkpoint: {args.resume}")

    # create trainer
    trainer = Trainer(
        logger=load_instance(config['trainer']['logger']),
        callbacks=load_instance(config['trainer']['callbacks']),
        max_epochs=config['trainer']['max_epochs'],
        precision='16-mixed'
    )

    if not args.test_only:
        # train the model
        trainer.fit(model, dataset)

    # test the model
    trainer.test(model, dataset)


if __name__ == "__main__":
    main()
