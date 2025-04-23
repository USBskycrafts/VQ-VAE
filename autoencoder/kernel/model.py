from typing import Any, Dict, List, Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.utils import make_grid
from autoencoder.utils import load_instance
from torchmetrics.image import (PeakSignalNoiseRatio as PSNR,
                                StructuralSimilarityIndexMeasure as SSIM)
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
        self.automatic_optimization = False

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

    def training_step(self, batch):
        x, y = batch
        x = x.to(self.device)  # 确保输入在正确的设备上
        y = y.to(self.device)  # 确保目标在正确的设备上'

        optimizers = self.optimizers()
        if isinstance(optimizers, list):
            g_optim, d_optim = optimizers
        else:
            raise RuntimeError('optimzier configuration error')
        qloss, xrec, perplexity, *_ = self(x)

        # autoencode
        self.toggle_optimizer(g_optim)
        g_optim.zero_grad()
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self._get_last_layer(), split="train")

        self.log("train/aeloss", aeloss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False,
                      logger=True, on_step=True, on_epoch=True)
        self.manual_backward(aeloss)
        g_optim.step()
        self.untoggle_optimizer(g_optim)

        # discriminator
        qloss, xrec, perplexity, *_ = self(x)
        self.toggle_optimizer(d_optim)
        d_optim.zero_grad()
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self._get_last_layer(), split="train")
        self.log("train/discloss", discloss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False,
                      logger=True, on_step=True, on_epoch=True)
        self.manual_backward(discloss)
        d_optim.step()
        self.untoggle_optimizer(d_optim)

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
        if isinstance(self.logger, TensorBoardLogger):
            with torch.no_grad():
                xrec = xrec.view(-1, 1, xrec.shape[-2], xrec.shape[-1])
                y = y.view(-1, 1, y.shape[-2], y.shape[-1])
                output_grid = make_grid(
                    xrec, nrow=8, normalize=True, scale_each=True)
                gt_grid = make_grid(y, nrow=8, normalize=True, scale_each=True)
                self.logger.experiment.add_image("val/reconstruction",
                                                 output_grid, self.global_step)
                self.logger.experiment.add_image("val/gt",
                                                 gt_grid, self.global_step)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        qloss, xrec, perplexity, *_ = self(x)
        psnr = PSNR(y.max() - y.min()).to(self.device)
        ssim = SSIM().to(self.device)

        self.log('test/PSNR', psnr(y, xrec), sync_dist=True)
        self.log('test/SSIM', ssim(y, xrec), sync_dist=True)

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
