import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time
from torch.amp import autocast, GradScaler


class ResNet50Bench(object):
    def __init__(self, gpu_device, cpu_device, epochs=5, batch_size=4, lr=0.001, data_size=1000, image_size=(3, 32, 32), num_classes=10, use_fp16=False):
        self.gpu_devices = gpu_device
        self.cpu_device = cpu_device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.data_size = data_size
        self.use_fp16 = use_fp16
        self.train_dataset = FakeDataset(size=data_size, image_size=image_size, num_classes=num_classes)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    def start(self):
        if self.gpu_devices is None:
            print("GPU is not available, only CPU will be benched.")
            print("DEBUG mode, skipping CPU bench.")
            # self._bench(self.cpu_device)
        else:
            print("GPU is available, both GPU and CPU will be benched.")
            print("DEBUG mode, skipping CPU bench.")
            self._bench(self.gpu_devices)
            # self._bench(self.cpu_device)

    def _bench(self, devices):
        main_device = devices[0]
        model = ResNet50().to(main_device)

        if len(self.gpu_devices) > 1:
            model = nn.DataParallel(model, device_ids=[device.index for device in devices])

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr)
        if main_device.type in ["xpu", "mps"]:
            GS_dev = "cuda"
        else:
            GS_dev = main_device.type
        scaler = GradScaler(device=GS_dev, enabled=self.use_fp16)

        total_step = len(self.train_loader)
        pre_load_start = time.time()
        data_preloaded = [(images.to(main_device), labels.to(main_device)) for images, labels in self.train_loader]
        pre_load_end = time.time()
        print(f"Pre-load completed on {main_device}. Time taken: {pre_load_end - pre_load_start:.2f} seconds.")

        start_time = time.time()
        for epoch in range(self.epochs):
            iters = len(self.train_loader)
            pbar = tqdm(total=iters, desc=f"Epoch: {epoch+1}/{self.epochs}", unit="it")
            for i, (images, labels) in enumerate(data_preloaded):
                # images = images.to(device)
                # labels = labels.to(device)

                with autocast(device_type=main_device.type, dtype=torch.float16, enabled=self.use_fp16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                pbar.update(1)
                pbar.set_postfix_str(f"Step {i+1}/{total_step}, Loss {loss.item():.4f}")

            pbar.close()
        end_time = time.time()
        time_usage = end_time - start_time
        basic_score = self.data_size / time_usage
        final_score = basic_score * (self.epochs / 10) * 100
        print(f"Training completed on {main_device}. Time taken: {time_usage:.2f} seconds. Score: {final_score:.0f}")


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000, image_size=(3, 32, 32), num_classes=10):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.data = torch.randn(size, *image_size)
        self.labels = torch.randint(0, num_classes, (size,))

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.size


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

# Example usage:
# model = ResNet50()
# print(model)
