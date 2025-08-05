"""Trains a GAN to generate Anti-Microbial Peptides."""

import argparse
import datetime
import logging
from pathlib import Path

import callbacks
import data_utils
import matplotlib.pyplot as plt
import model
import pandas as pd
import tensorflow as tf

import CFGAN.utils.analysor_predictor as analysor_predictor


def get_parser():
    parser = argparse.ArgumentParser(
        description="Trains a CFGAN to generate Anti-microbial Peptides (AMPs).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of samples per training batch.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5_00,
        help="Number of times the data set if shown to the GAN.",
    )
    parser.add_argument(
        "--model_type",
        type=str.lower,
        default="gan",
        choices={"gan", "wgan"},
        help="GAN structure to use.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restarts training from the most recently saved checkpoints.",
    )
    parser.add_argument(
        "-t",
        "--tag",
        type=str.lower,
        default="",
        help="An identifier used to distinguish simultaneous experiments.",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Amount of terminal output.",
    )

    return parser


def main(
    batch_size: int = 128,
    epochs: int = 5_00,
    model_type: str = "gan",
    tag: str = "",
    verbose: int = 0,
    reg_strength: float = 100000000.0,
    **kwargs,
):
    data = data_utils.get_train_data(batch_size)
    n_batch = len(data)
    gan = model.CFGAN(
        model.amp_discriminator(final_activation="sigmoid"),
        model.amp_generator(),
        n_batch=n_batch,
        reg_strength=reg_strength,
    )
    
    gan.compile()

    loss_file = Path(f"D:\\reoccur\\CFGAN\\results\\losses_{datetime.datetime.now().date()}{tag}.csv")
    cp_dir = Path(f"D:\\reoccur\\CFGAN\\models\\amp_gan_{str(datetime.datetime.now().date())}{tag}")
    cp_dir.mkdir(exist_ok=True, parents=True)
    try:
        gan.fit(
            data,
            epochs=epochs,
            callbacks=[
                callbacks.AMPQualityLogger(data),
                callbacks.ModelCheckpoint(
                    str(cp_dir / (f"{model_type}" + "_{epoch:04d}"))
                ),
                callbacks.RegularizationScheduler(
                    lambda epoch, reg: (1 - (epoch / (epochs - 1))) * reg_strength
                ),
                tf.keras.callbacks.CSVLogger(loss_file),
                callbacks.TerminateOnNaN(),
            ],
            verbose=verbose,
        )
    except KeyboardInterrupt:
        pass

    make_loss_plot(loss_file)


def make_loss_plot(loss_file: Path):
    df = pd.read_csv(loss_file, index_col=0)
    df.index += 1
    fig, axes = plt.subplots(len(df.columns), 1, sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(df.index[:-100], df.iloc[:-100, i], label=df.columns[i])
        ax.legend()

    best_r2_score = df["g_r2_score"].max()
    best_epoch = df.index[df["g_r2_score"].argmax()]
    fig.suptitle(f"Best R2 Score of {best_r2_score:0.4f}\n@ Epoch {best_epoch}")
    plt.savefig(loss_file.with_suffix(".png"))
    plt.close()


if __name__ == "__main__":
    args = vars(get_parser().parse_args())

    logger = logging.getLogger()

    if args["tag"]:
        args["tag"] = f"_{args['tag']}"
    handler = logging.FileHandler(
        f"D:\\reoccur\\CFGAN\\results\\train_{datetime.datetime.now().date()}{args['tag']}.log"
    )
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


    main(**args)
