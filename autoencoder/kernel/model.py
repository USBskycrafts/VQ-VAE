from typing import Any, Dict, List, Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn


from autoencoder.utils import load_instance
from autoencoder.kernel.modules import Encoder, Decoder, VectorQuantizer


class VQVAE(pl.LightningModule):
    def __init__(self, *,
                 backbone_config: Dict,
                 quantizer_config: Dict,
                 loss_config: Dict,
                 ckpt_path=None,
                 ignore_keys=[],
                 learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(**backbone_config)
        self.decoder = Decoder(**backbone_config)
        self.quantizer = VectorQuantizer(**quantizer_config)
        self.loss = load_instance(loss_config)
        self.learning_rate = learning_rate

        if ckpt_path is not None:
            self._init_from_ckpt(ckpt_path, ignore_keys)

    def forward(self, x: torch.Tensor) -> Tuple[Any, ...]:
        # Encoder
        z_e = self.encoder(x)
        # Quantizer
        q_loss, z_q, perplexity, *others = self.quantizer(z_e)
        # Decoder
        x_recon = self.decoder(z_q)

        return q_loss, x_recon, perplexity, *others

    def training_step(self, batch, batch_idx: int, optimizer_idx: int):
        x, y = batch
        x = x.to(self.device)  # 确保输入在正确的设备上
        y = y.to(self.device)  # 确保目标在正确的设备上
        assert x.shape == y.shape, f"Input and target shapes mismatch: {x.shape} != {y.shape}"
        qloss, xrec, perplexity, *_ = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self._get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False,
                          logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self._get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True,
                     logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False,
                          logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        qloss, xrec, perplexity, *_ = self(x)
        aeloss, log_dict_ae = self.loss(qloss, y, xrec, 0, self.global_step,
                                        last_layer=self._get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self._get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss, sync_dist=True)
        self.log("val/aeloss", aeloss, sync_dist=True)
        self.log("val/discloss", discloss, sync_dist=True)
        self.log("val/perplexity", perplexity, sync_dist=True)
        self.log_dict(log_dict_ae, sync_dist=True)
        self.log_dict(log_dict_disc, sync_dist=True)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.AdamW(list(self.encoder.parameters()) +
                                   list(self.decoder.parameters()) +
                                   list(self.quantizer.parameters()),
                                   lr=lr)
        opt_disc = torch.optim.AdamW(self.loss.parameters(),
                                     lr=lr)
        return [opt_ae, opt_disc], []

    # -------------------------------------------------------------------
    # inner dirty work

    def _get_last_layer(self):
        return self.decoder.conv_out.weight

    def _init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    # -------------------------------------------------------------------
