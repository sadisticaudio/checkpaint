#/bin/bash

from python cli - exec("for i in range(10): print(i)")

## build all versions

MATRIX

auditwheel -v repair \
	--plat manylinux_2_17_x86_64 \
	--exclude libc10-56a79b4f.so \
	--exclude libc10_cuda \
	--exclude libcublas \
	--exclude libcublasLt.so.11 \
	--exclude libcuda \
	--exclude libcudart.so.11.0 \
	--exclude libcudart-d0da41ae \
	--exclude libcudnn \
	--exclude libgomp \
	--exclude libnvrtc \
	--exclude libnvToolsExt \
	--exclude libnvToolsExt-847d78f2 \
	--exclude libshm \
	--exclude libtorch.so \
	--exclude libtorch_cpu.so \
	--exclude libtorch_cuda.so \
	--exclude libtorch_python.so \
	--only-plat \
	checkpaint-0.0.1-cp310-cp310-linux_x86_64.whl


g++ -I"${theTorchRoot}/include;${theTorchRoot}/include/torch/csrc/api/include"-M -o ./modules/embedding.deps ./modules/embedding.cpp

## PyTorch download links
https://download.pytorch.org/libtorch/cu121
https://download.pytorch.org/libtorch/cu118
https://download.pytorch.org/whl/torch/

