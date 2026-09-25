# CUHK-X Small Model Track — submission inference image.
#
# Build from the repository root so Docker and native inference share the same
# Python package and the same single checkpoint:
#
#   docker build -f docker/Dockerfile -t cuhk-small-infer:latest .
#   docker run --rm --gpus all \
#     -v /path/to/testing_dir:/data:ro \
#     -v "$(pwd)":/out \
#     cuhk-small-infer:latest

FROM python:3.13-slim-bookworm

# OpenCV/ultralytics runtime libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libgl1 \
        libx11-6 \
        libx11-xcb1 \
        libxext6 \
        libxfixes3 \
        libxi6 \
        libxrender1 \
        libxrandr2 \
        libxinerama1 \
        libxcursor1 \
        libxft2 \
        libxcb1 \
        libsm6 \
        libxxf86vm1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/share/zoneinfo/Etc/UTC /etc/localtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/tmp

WORKDIR /app

# Keep the large CUDA-enabled torch layer independent from application code.
RUN pip install \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        "torch==2.14.0" \
        "torchvision==0.29.0"

# Remaining pinned runtime dependencies.
COPY docker/requirements.txt /app/docker/requirements.txt
RUN pip install -r /app/docker/requirements.txt

# One canonical source tree and one canonical checkpoint. The native CLI uses
# these same files from the host checkout; Docker copies them into the image.
COPY pyproject.toml README.md /app/
COPY code/ /app/code/
RUN pip install --no-deps -e /app \
    && python -c "from ultralytics import SETTINGS; import sota_release_infer2; print('import graph OK')"

# The Python CLI auto-discovers the SM_test_XXXX root under /data and writes
# the CSV to the host's current directory mounted at /out.
CMD ["samhar", "/data", "/out/submission.csv"]
