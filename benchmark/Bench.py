import torch
import sys
from benchmark.bench.cnn_bench import CNNBench


class Bench(object):
    def __init__(self, method="cnn"):
        self.gpu_device = self._get_gpu_device()
        self.cpu_device = self._get_cpu_device()
        self.backend = self._load_backend(method=method)

    def start(self):
        self.backend.start()

    @staticmethod
    def _get_gpu_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return None

    @staticmethod
    def _get_cpu_device():
        return torch.device("cpu")

    def _load_backend(self, method):
        if method == "cnn":
            return CNNBench(gpu_device=self.gpu_device, cpu_device=self.cpu_device, data_size=500000, batch_size=1024)

