# Runtime shim for the pinned vLLM 0.32.0 cu129 image, mounted on PYTHONPATH.
# The image ships torchcodec 0.17.0 built against CUDA 13 (needs libnvrtc.so.13),
# so its import raises OSError, which vllm/multimodal/media/audio.py does not catch.
# Raising ImportError takes vLLM's existing "torchcodec unavailable" path; audio
# decoding is unused by this text-only workload. The image itself is unmodified.
raise ImportError("torchcodec disabled by bonfire compat shim (CUDA 13 build in a cu129 image)")
