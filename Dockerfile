# Thirty Two on a CPU (also the Hugging Face Spaces image: Docker SDK, port 7860, user 1000).
# The weights are not in the image: the app downloads cusp_assets.zip from the GitHub release at first start.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/user \
    THIRTYTWO_ASSETS=/home/user/assets \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt \
 && pip uninstall -y opencv-python \
 && pip install --force-reinstall --no-deps opencv-python-headless==5.0.0.93

COPY --chown=user dentex_utils.py train_arch.py ./
COPY --chown=user thirtytwo ./thirtytwo

USER user
EXPOSE 7860
CMD ["uvicorn", "thirtytwo.server:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log"]
