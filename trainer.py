from medsrgan import Generator, Discriminator, FeatureExtractor
from dataset import GAN_Data
from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as ttf
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable


device = "cuda" if torch.cuda.is_available() else "cpu"
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
path = r"D:/Desktop/Medical Imaging/MRI_512/"
path_list = os.listdir(path)


gen = Generator().to(device)
disc = Discriminator().to(device)
feature_extractor = FeatureExtractor().to(device)
feature_extractor.eval()

transforms = ttf.Compose(
    [
        ttf.RandomHorizontalFlip(0.5),
        ttf.RandomVerticalFlip(0.5),
        ttf.RandomRotation((-15, 15)),
    ]
)

dataset = GAN_Data(path_list, transforms)
train_dl = DataLoader(dataset, 1, True)

# plt.imshow(dataset[0][1].squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
# plt.show()

optimizer_G = optim.Adam(gen.parameters(), lr=1e-3, weight_decay=1e-5)
optimizer_D = optim.Adam(disc.parameters(), lr=1e-4, weight_decay=1e-5)
loss_function = torch.nn.L1Loss().to(device)
gan_loss = torch.nn.BCEWithLogitsLoss().to(device)
scaler = torch.cuda.amp.GradScaler()


def fit(
    gen,
    disc,
    feature_extractor,
    train_dl,
    epochs,
    optimizer_G,
    optimizer_D,
    scaler,
    loss_function,
    gan_loss,
):

    t_loss_G, t_loss_D = [], []

    for epoch in tqdm(range(epochs)):
        e_loss_G, e_loss_D = [], []

        for data in train_dl:
            hr_img, lr_img = data

            valid = Variable(Tensor(np.ones((1, 2))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((1, 2))), requires_grad=False)

            with torch.cuda.amp.autocast():

                # Train Generator

                pred_hr = gen(lr_img)

                content_loss = loss_function(pred_hr, hr_img)

                pred_features = feature_extractor(pred_hr)
                hr_features = feature_extractor(hr_img)

                feature_loss = 0.0

                for pred_feature, hr_feature in zip(pred_features, hr_features):
                    feature_loss += loss_function(pred_feature, hr_feature)

                pred_real = disc(hr_img.detach(), lr_img)
                pred_fake = disc(pred_hr, lr_img)

                gan_loss_num = gan_loss(
                    pred_fake - pred_real.mean(0, keepdim=True), valid
                )

                loss_G = content_loss * 0.1 + feature_loss * 0.1 + gan_loss_num

                optimizer_G.zero_grad()
                scaler.scale(loss_G).backward()
                scaler.step(optimizer_G)
                scaler.update()
                e_loss_G.append(loss_G)

                # Train Discriminator

                pred_real = disc(hr_img, lr_img)
                pred_fake = disc(pred_hr.detach(), lr_img)

                loss_real = gan_loss(pred_real - pred_fake.mean(0, keepdim=True), valid)
                loss_fake = gan_loss(pred_fake - pred_real.mean(0, keepdim=True), fake)

                loss_real_num = gan_loss(pred_real, valid)
                loss_fake_num = gan_loss(pred_fake, fake)

                loss_D = ((loss_real + loss_fake) / 2) + (
                    (loss_real_num + loss_fake_num) / 2
                )

                optimizer_D.zero_grad()
                scaler.scale(loss_D).backward()
                scaler.step(optimizer_D)
                scaler.update()
                e_loss_D.append(loss_D)

        t_loss_D.append(sum(e_loss_D) / len(e_loss_D))
        t_loss_G.append(sum(e_loss_G) / len(e_loss_G))

        print(
            f"{epoch+1}/{epochs} -- Gen Loss: {sum(t_loss_G) / len(t_loss_G)} -- Disc Loss: {sum(t_loss_D) / len(t_loss_D)}"
        )

        torch.save(gen, "./gen_{epoch}")
        torch.save(disc, "./disc_{epoch}")

    return t_loss_G, t_loss_D
 
