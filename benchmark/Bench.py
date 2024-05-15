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

    @staticmethod
    def _get_cuda_memory_size():
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties("cuda")
            print(f"Found cuda device: {props.name}, CUDA architecture: {props.major}.{props.minor}\nFound {props.total_memory / 1024 / 1024:.2f} MB CUDA memory available.")
            return props.total_memory
        else:
            return 0

    def _load_backend(self, method):
        if method == "cnn":
            cuda_memory_size = self._get_cuda_memory_size()
            data_size = int(int((cuda_memory_size / 12296)/100) * 100 * 0.7)
            print(f"Set data size to {data_size} images, total memory size: {data_size * 12296 / 1024 / 1024:.2f} MB")
            return CNNBench(gpu_device=self.gpu_device, cpu_device=self.cpu_device, data_size=data_size, batch_size=2048, epochs=10)

