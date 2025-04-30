import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, scale=1.0):
        super(ContrastiveLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.scale = scale

    def forward(self, seg_features, image_features):
        """
        Compute the contrastive loss between segmentation features and image features.
        
        Args:
            seg_features (torch.Tensor): Segmentation features of shape (N, C, H, W).
            image_features (torch.Tensor): Image features of shape (N, C, H, W).
        
        Returns:
            torch.Tensor: Contrastive loss.
        """
        return self._cal_contrast_loss(seg_features, image_features) 
        
    def _cal_contrast_loss(self, seg_features, image_features):
        seg_features = seg_features.reshape(-1, seg_features.shape[1], seg_features.shape[2] * seg_features.shape[3])
        image_features = image_features.reshape(-1, image_features.shape[1], image_features.shape[2] * image_features.shape[3])
        image_features = image_features / torch.norm(image_features, dim=1, keepdim=True)
        seg_features = seg_features / torch.norm(seg_features, dim=1, keepdim=True) 
        
        # [B, H*W, H*W]
        logits_per_image = torch.matmul(image_features.permute(0, 2, 1), seg_features) * self.scale
        logits_per_seg = logits_per_image.permute(0, 2, 1)

        label = torch.arange(0, seg_features.shape[-1], device=seg_features.device)
        label =  label.unsqueeze(0).repeat(seg_features.shape[0], 1)
        loss_image = self.cross_entropy_loss(logits_per_image, label)
        loss_seg = self.cross_entropy_loss(logits_per_seg, label)
        return loss_image + loss_seg 