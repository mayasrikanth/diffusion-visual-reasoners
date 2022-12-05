import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from clip import clip

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class CLIPRPrecision(pl.LightningModule):
    def __init__(self, init_lr=1e-5):
        super().__init__()
        clip_model, preprocess = clip.load('RN101', jit=False)
        clip_model = clip_model.float()
        self.preprocess = preprocess
        self.learning_rate = 1e-5

        # visual
        self.visual_frozen = nn.Sequential(
            clip_model.visual.conv1,
            clip_model.visual.bn1,
            clip_model.visual.relu,
            clip_model.visual.conv2,
            clip_model.visual.bn2,
            clip_model.visual.relu,
            clip_model.visual.conv3,
            clip_model.visual.bn3,
            clip_model.visual.relu,
            clip_model.visual.avgpool,
            clip_model.visual.layer1,
            clip_model.visual.layer2,
            clip_model.visual.layer3,
        ).eval().requires_grad_(False)

        self.attn_pool = clip_model.visual.attnpool
        self.layer4 = clip_model.visual.layer4

        # textual
        self.token_embedding_frozen = clip_model.token_embedding.eval().requires_grad_(False)
        self.positional_embedding_frozen = clip_model.positional_embedding.requires_grad_(False)
        self.transformer_frozen = nn.Sequential(
            *clip_model.transformer.resblocks[:-1]
        ).eval().requires_grad_(False)

        self.transformer_last_block = clip_model.transformer.resblocks[-1]

        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.logit_scale = clip_model.logit_scale

    def encode_image(self, image):
        with torch.no_grad():
            x = self.visual_frozen(image)
        x = self.layer4(x)
        x = self.attn_pool(x)

        return x

    def encode_text(self, text):
        with torch.no_grad():
            x = self.token_embedding_frozen(text)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding_frozen
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer_frozen(x)

        x = self.transformer_last_block(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def unfreeze(self):
        self.attn_pool.requires_grad_(True)
        self.layer4.requires_grad_(True)

        self.transformer_last_block.requires_grad_(True)
        self.ln_final.requires_grad_(True)
        self.text_projection.requires_grad_(True)
        self.logit_scale.requires_grad_(True)

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        return image_features, text_features

    def training_step(self, batch, batch_idx):
        # image, text = batch
        image, text = batch.values()    # support custom winoground data
        image = image.permute(0, 3, 1, 2)
        image = image.float()
        # import pdb; pdb.set_trace()
        bs = image.size(0)

        image_features, text_features = self(image, text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()


        # corrected loss
        im_sim = image_features @ image_features.T
        txt_sim = text_features @ text_features.T
        targets = F.softmax((im_sim + txt_sim)/2 * logit_scale, dim=-1)
        logits = logit_scale * text_features @ image_features.T
        txt_loss = cross_entropy(logits, targets).mean()
        im_loss = cross_entropy(logits.T, targets.T).mean()

        label = torch.arange(bs).long()
        label = label.to(image.device)

        # loss_i = F.cross_entropy(logits_per_image, label)
        # loss_t = F.cross_entropy(logits_per_text, label)

        # loss = (loss_i + loss_t) / 2
        loss = (txt_loss + im_loss) / 2
        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.AdamW(list(self.attn_pool.parameters()) +
                                list(self.layer4.parameters()) +
                                list(self.transformer_last_block.parameters()) +
                                list(self.ln_final.parameters()) +
                                [self.text_projection, self.logit_scale],
                                lr=lr)
        return opt

