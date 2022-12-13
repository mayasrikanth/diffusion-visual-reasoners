import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import datasets
from torch.utils.data import DataLoader
from models.clip_r_precision import CLIPRPrecision
import torch
from clip import clip

#TRAIN_DATA_DIR = '/home/jasonlin/repos/datasets/t2i_benchmark/winoground/hf/winoground-hf-arrow-label'
TRAIN_DATA_DIR = '/home/mayashar/Desktop/gqa_winogrand_val_clipR'
TRAIN_PARQUET = '/home/jasonlin/repos/datasets/t2i_benchmark/winoground/hf/winoground-hf-train.parquet'
#CLIP_PATH = '/home/jasonlin/repos/comp-t2i-dataset/clip_weights/winoground/winoground_shape.pt'
CLIP_PATH = '/home/mayashar/Desktop/C-CUB/C_CUB_shape.pt'

def train(train_data):
    # load the model
    clip_model = CLIPRPrecision()
    sd = torch.load(CLIP_PATH, map_location="cpu")["state_dict"]
    missing, unexpected = clip_model.load_state_dict(sd, strict=False)
    clip_model = clip_model.cuda()
    # import pdb; pdb.set_trace()
    # call tune to find the lr
    trainer.tune(clip_model, train_dataloaders=train_data)  # ValueError: An invalid dataloader was passed to `Trainer.fit(train_dataloaders=...)`. Either pass the dataloader to the `.fit()
    trainer.fit(model=clip_model, train_dataloaders=train_data)


wandb_logger = WandbLogger(name='winoground_clip_ft_debug',project='diffusion-reasoner')

# preprocess data
def transforms(examples):
    print("EXAMPLES")
    print(examples)
    # reshape images to 224, 224 as clip expects image HW = 224, token length = 77 
    examples["image"] = [image.convert("RGB").resize((224,224)) for image in examples["image"]]
    examples["label"] = clip.tokenize(examples["label"])
    return examples

hf_dataset = datasets.load_from_disk(TRAIN_DATA_DIR).with_format("torch")
print("TYPE OF DATASET: ", type(hf_dataset))
print("DATA: ", hf_dataset[0])
hf_dataset = hf_dataset.map(transforms, batched=True)
print("ANOTHER TYPING: ", type(hf_dataset))
# import pdb; pdb.set_trace()
# wino_train_loader = DataLoader(hf_dataset['train'].with_format("torch"), batch_size=512)

wino_train_loader = DataLoader(hf_dataset.with_format("torch"), batch_size=512)

# dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
# train_loader = utils.data.DataLoader(dataset)
#Trainer(callbacks=[CheckpointEveryNSteps()])
trainer = pl.Trainer(max_epochs=15, accelerator="gpu", auto_lr_find=True, 
    log_every_n_steps=5, logger=wandb_logger, callbacks=[ModelCheckpoint(dirpath='./',every_n_epochs=2)])   # limit_train_batches=1, logger=wandb_logger

# import pdb; pdb.set_trace()
train(wino_train_loader)
