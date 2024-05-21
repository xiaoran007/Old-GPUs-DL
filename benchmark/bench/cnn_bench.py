import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time


class CNNBench(object):
    def __init__(self, gpu_device, cpu_device, epochs=5, batch_size=4, lr=0.001, data_size=1000, image_size=(3, 32, 32), num_classes=10):
        self.gpu_device = gpu_device
        self.cpu_device = cpu_device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.data_size = data_size
        self.train_dataset = FakeDataset(size=data_size, image_size=image_size, num_classes=num_classes)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

    def start(self):
        if self.gpu_device is None:
            print("GPU is not available, only CPU will be benched.")
            self._bench(self.cpu_device)
        else:
            print("GPU is available, both GPU and CPU will be benched.")
            print("DEBUG mode, skipping CPU bench.")
            self._bench(self.gpu_device)
            # self._bench(self.cpu_device)

    def _bench(self, device):
        model = CNN().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr)

        total_step = len(self.train_loader)
        pre_load_start = time.time()
        data_preloaded = [(images.to(device), labels.to(device)) for images, labels in self.train_loader]
        pre_load_end = time.time()
        print(f"Pre-load completed on {device}. Time taken: {pre_load_end - pre_load_start:.2f} seconds.")

        start_time = time.time()
        for epoch in range(self.epochs):
            iters = len(self.train_loader)
            pbar = tqdm(total=iters, desc=f"Epoch: {epoch+1}/{self.epochs}", unit="it")
            for i, (images, labels) in enumerate(data_preloaded):
                # images = images.to(device)
                # labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix_str(f"Step {i+1}/{total_step}, Loss {loss.item():.4f}")

            pbar.close()
        end_time = time.time()
        time_usage = end_time - start_time
        basic_score = self.data_size / time_usage
        final_score = basic_score * (self.epochs / 10)
        print(f"Training completed on {device}. Time taken: {time_usage:.2f} seconds. Score: {final_score:.2f}")


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


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

