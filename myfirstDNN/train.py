import torch
torch.set_float32_matmul_precision('high')

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from app.ml.dataset import FaceDataset, get_default_transforms
from app.ml.model import FaceEncoder
from app.ml.loss import ArcFaceLoss


class FaceRecognitionModel(pl.LightningModule):
    def __init__(self, num_classes, embedding_size=512, lr=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = FaceEncoder(embedding_size=embedding_size)
        self.criterion = ArcFaceLoss(num_classes, embedding_size, s=8.0, m=0.5)
        self.lr = lr

    def forward(self, x):
        return self.encoder(x)

    def on_train_epoch_start(self):
        """Повышаем scale по фазам"""
        epoch = self.current_epoch
        if epoch < 15:
            self.criterion.set_scale(8.0)
        elif epoch < 25:
            self.criterion.set_scale(12.0)
        elif epoch < 32:
            self.criterion.set_scale(16.0)
        else:
            self.criterion.set_scale(24.0)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        embeddings = self.encoder(images)
        loss = self.criterion(embeddings, labels)

        with torch.no_grad():
            sim = torch.matmul(embeddings, embeddings.t())
            mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)

            # Межклассовое сходство (разные люди)
            diff_class = (labels.unsqueeze(0) != labels.unsqueeze(1)) & mask
            inter_sim = sim[diff_class].mean() if diff_class.any() else torch.tensor(0.0)

            # Внутриклассовое сходство (один человек)
            same_class = (labels.unsqueeze(0) == labels.unsqueeze(1)) & mask
            intra_sim = sim[same_class].mean() if same_class.any() else torch.tensor(0.0)

            # Угол к своему центроиду
            norm_weight = nn.functional.normalize(self.criterion.weight, dim=1)
            cosine_own = torch.sum(embeddings * norm_weight[labels], dim=1)
            theta_own = torch.acos(torch.clamp(cosine_own, -1.0 + 1e-7, 1.0 - 1e-7)).mean()
            theta_deg = torch.rad2deg(theta_own)

        self.log('loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('s', self.criterion.s, prog_bar=True, on_step=True, on_epoch=True)
        self.log('inter', inter_sim, prog_bar=True, on_step=True, on_epoch=True)
        self.log('intra', intra_sim, prog_bar=True, on_step=True, on_epoch=True)
        self.log('θ°', theta_deg, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=0.01)
        return [optimizer], [scheduler]


def main():
    DATA_DIR = 'data/processed'
    BATCH_SIZE = 200
    EPOCHS = 40

    print("Загрузка датасета...")
    dataset = FaceDataset(root_dir=DATA_DIR, transform=get_default_transforms(mode='train'))
    num_classes = len(dataset.classes)
    print(f"Классов: {num_classes}, Фото: {len(dataset)}")

    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=6, pin_memory=True, persistent_workers=True
    )

    #model = FaceRecognitionModel(num_classes=num_classes)
    # Загружаем чекпоинт
    checkpoints = sorted(Path('app/ml/checkpoints').glob('face-v3-*.ckpt'))
    checkpoint_path = str(checkpoints[-1]) if checkpoints else None
    print(f"Загружаем чекпоинт: {checkpoint_path}")
    model = FaceRecognitionModel.load_from_checkpoint(
        checkpoint_path,
        num_classes=num_classes
    )


    checkpoint_callback = ModelCheckpoint(
        dirpath='app/ml/checkpoints',
        filename='face-v3-{epoch:02d}-{loss:.3f}',
        save_top_k=3, monitor='loss', mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='gpu', devices=1,
        precision='16-mixed',
        gradient_clip_val=2.0,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader)
    torch.save(model.encoder.state_dict(), 'app/ml/checkpoints/face_encoder.pt')
    print("Модель сохранена!")


if __name__ == "__main__":
    main()