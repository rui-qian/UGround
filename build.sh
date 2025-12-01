# Credit: Chuanhang Deng, © 2025 Fudan University. All rights reserved.
#!/bin/bash
# 1. curl -O https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
# 2. bash Anaconda3-2025.06-0-Linux-x86_64.sh
# 3. conda create -n uground python=3.9
# 4. conda activate uground
# 5. chmod +x build.sh
# 6. ./build.sh
# 7. wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
# 8. pip install flash_attn-2.6.3+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl


set -e

pip install --upgrade pip setuptools wheel

pip install numpy==1.24.2
pip install cmake==3.30.3 Cython==3.0.11 ninja==1.11.1.1 pybind11==2.11.1

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 triton==2.0.0 --index-url https://download.pytorch.org/whl/cu118

pip install einops==0.4.1

pip install peft==0.4.0

pip install huggingface-hub==0.25.0
pip install safetensors==0.4.5
pip install tokenizers==0.13.3
pip install transformers==4.31.0
pip install accelerate==0.34.2
pip install deepspeed==0.10.3
pip install spacy==3.6.1

pip install \
    absl-py==2.1.0 \
    aiofiles==23.2.1 \
    altair==5.5.0 \
    annotated-types==0.7.0 \
    anyio==4.9.0 \
    appdirs==1.4.4 \
    attrs==25.3.0 \
    autocommand==2.2.2 \
    backports.tarfile==1.2.0 \
    blis==0.7.11 \
    cachetools==5.5.0 \
    catalogue==2.0.10 \
    certifi==2024.8.30 \
    charset-normalizer==3.3.2 \
    click==8.1.7 \
    confection==0.1.5 \
    contourpy==1.3.0 \
    cycler==0.12.1 \
    cymem==2.0.8 \
    docker-pycreds==0.4.0 \
    exceptiongroup==1.3.0 \
    fastapi==0.115.12 \
    ffmpy==0.6.0 \
    filelock==3.16.1 \
    fonttools==4.53.1 \
    fsspec==2024.9.0 \
    gitdb==4.0.11 \
    GitPython==3.1.43 \
    google-auth==2.35.0 \
    google-auth-oauthlib==1.2.1 \
    gradio==3.50.0 \
    gradio_client==0.6.1 \
    grpcio==1.66.1 \
    h11==0.16.0 \
    hjson==3.1.0 \
    httpcore==1.0.9 \
    httpx==0.28.1 \
    idna==3.10 \
    imageio==2.35.1 \
    importlib_metadata==8.5.0 \
    importlib_resources==6.4.5 \
    inflect==7.3.1 \
    jaraco.collections==5.1.0 \
    jaraco.context==5.3.0 \
    jaraco.functools==4.0.1 \
    jaraco.text==3.12.1 \
    Jinja2==3.1.4 \
    jsonschema==4.24.0 \
    jsonschema-specifications==2025.4.1 \
    kiwisolver==1.4.7 \
    langcodes==3.4.0 \
    language_data==1.2.0 \
    lazy_loader==0.4 \
    lit==18.1.8 \
    marisa-trie==1.2.0 \
    Markdown==3.7 \
    markdown-it-py==3.0.0 \
    MarkupSafe==2.1.5 \
    matplotlib==3.9.2 \
    mdurl==0.1.2 \
    more-itertools==10.3.0 \
    mpmath==1.3.0 \
    murmurhash==1.0.10 \
    narwhals==1.41.1 \
    networkx==3.2.1 \
    oauthlib==3.2.2 \
    opencv-python==4.8.0.74 \
    opencv-python-headless==4.8.0.74 \
    orjson==3.10.18 \
    packaging==24.1 \
    pandas==2.3.0 \
    pathlib_abc==0.1.1 \
    pathy==0.11.0 \
    Pillow==10.4.0 \
    platformdirs==4.2.2 \
    preshed==3.0.9 \
    protobuf==4.23.4 \
    psutil==6.0.0 \
    py-cpuinfo==9.0.0 \
    pyasn1==0.6.1 \
    pyasn1_modules==0.4.1 \
    pycocotools==2.0.8 \
    pydantic==1.10.18 \
    pydantic_core==2.33.2 \
    pydub==0.25.1 \
    Pygments==2.19.1 \
    pyparsing==3.1.4 \
    python-dateutil==2.9.0.post0 \
    python-multipart==0.0.20 \
    pytz==2025.2 \
    PyWavelets==1.6.0 \
    PyYAML==6.0.2 \
    referencing==0.36.2 \
    regex==2024.9.11 \
    requests==2.32.3 \
    requests-oauthlib==2.0.0 \
    rich==14.0.0 \
    rpds-py==0.25.1 \
    rsa==4.9 \
    ruff==0.11.13 \
    scikit-image==0.21.0 \
    scipy==1.13.1 \
    semantic-version==2.10.0 \
    sentencepiece==0.1.99 \
    sentry-sdk==2.14.0 \
    setproctitle==1.3.3 \
    shellingham==1.5.4 \
    six==1.16.0 \
    smart-open==6.4.0 \
    smmap==5.0.1 \
    sniffio==1.3.1 \
    spacy-legacy==3.0.12 \
    spacy-loggers==1.0.5 \
    srsly==2.4.8 \
    starlette==0.46.2 \
    sympy==1.13.3 \
    tensorboard==2.15.1 \
    tensorboard-data-server==0.7.2 \
    termcolor==2.5.0 \
    thinc==8.1.12 \
    tifffile==2024.8.30 \
    tomli==2.0.1 \
    tomlkit==0.12.0 \
    tqdm==4.64.1 \
    typeguard==4.3.0 \
    typer==0.16.0 \
    typing-inspection==0.4.1 \
    typing_extensions==4.12.2 \
    tzdata==2025.2 \
    urllib3==2.2.3 \
    uvicorn==0.34.3 \
    wandb==0.16.4 \
    wasabi==1.1.3 \
    websockets==11.0.3 \
    Werkzeug==3.0.4 \
    zipp==3.20.2
