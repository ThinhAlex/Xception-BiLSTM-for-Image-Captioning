import torch
from torchvision import transforms
import model
import pytorch_lightning as pl
from dataset import data_loader
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor


torch.manual_seed(42)  
# Hyperparameters
EMBEDDEDING_SIZE = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 1
EPOCH = 50

torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

transform = transforms.Compose(
    [
        transforms.Resize((356, 356)),
        transforms.RandomCrop((320, 320)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_loader, dataset = data_loader(
    root_folder="flickr8k/images",
    annotation_file="flickr8k/captions.txt",
    transform=transform,
    num_workers=8,
    batch_size=64,
)

model = model.CaptionModel(EMBEDDEDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, dataset)

for name, param in model.encoder.xception.named_parameters():
    if "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

if __name__ == "__main__":
    logger = TensorBoardLogger("tb_logs_8k", name = "caption_model_xception")
    trainer = pl.Trainer(
        logger = logger,
        gradient_clip_val=2,
        gradient_clip_algorithm="norm",
        accelerator="gpu", 
        devices=1,
        max_epochs = EPOCH,
        precision = "16-mixed",
        callbacks = [
                    EarlyStopping(monitor = "train_bleu_epoch", patience = 4, verbose = 1, mode="max"),
                    LearningRateMonitor(logging_interval='epoch'),
                    ]
    )
    
    trainer.fit(model, train_loader)
    trainer.test(model, train_loader)
