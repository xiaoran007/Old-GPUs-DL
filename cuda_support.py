from pathlib import Path
import re
from typing import Sequence, Tuple

from torch.utils.cpp_extension import include_paths as torch_include_paths


def get_supported_cuda_archs() -> Sequence[Tuple[int, int]]:
    cuda_config_h = Path(torch_include_paths()[0]) / "ATen" / "cuda" / "CUDAConfig.h"
    if not cuda_config_h.exists():
        return []

    nvcc_flags_re = re.compile(r'^#define NVCC_FLAGS_EXTRA "([^"]+)"', flags=re.MULTILINE)
    match = nvcc_flags_re.search(cuda_config_h.read_text())
    if match is None:
        return []
    nvcc_flags = match.group(1)

    compute_arch_re = re.compile(r"-gencode;arch=compute_(\d+)")
    cuda_archs = []
    for arch in compute_arch_re.findall(nvcc_flags):
        if divmod(int(arch), 10) in cuda_archs:
            continue
        cuda_archs.append(divmod(int(arch), 10))

    return sorted(cuda_archs)


def main():
    compute_capabilities_str = ', '.join(f'{major}.{minor}' for major, minor in get_supported_cuda_archs())
    if len(compute_capabilities_str):
        print(f"Found supported CUDA compute capabilities: {compute_capabilities_str}")
    else:
        print("No supported CUDA compute capabilities found.")


if __name__ == '__main__':
    main()
