import torch
import torchvision

import skimage.data

img = skimage.transform.resize(skimage.data.camera(), (64, 64))

model = torchvision.models.resnet50()
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.eval()

torch.save(model, "resnet50")

import time
import numpy
import tqdm
import joblib


def do():
    with torch.no_grad():
        return model(torch.tensor(numpy.array([[img]]), dtype=torch.float32)).numpy().flatten()


start = time.time()
res = [joblib.Parallel(n_jobs=-1)(joblib.delayed(do)() for _ in tqdm.trange(200))]
end = time.time()
print(end - start)
