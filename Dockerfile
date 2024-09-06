# Configure image
ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}-slim
ARG PYTHON_VERSION
ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN echo "source /opt/venv/bin/activate" >> /root/.bashrc

RUN useradd -m -u 1000 user

# Install LeRobot
RUN git clone --branch visualize_all_datastes https://github.com/mishig25/lerobot.git /lerobot
WORKDIR /lerobot
RUN pip install --upgrade --no-cache-dir pip
RUN pip install --no-cache-dir "." \
    --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir flask

COPY --chown=user . /lerobot
CMD ["python", "lerobot/scripts/visualize_dataset_html.py", "--host", "0.0.0.0", "--port", "7860"]
