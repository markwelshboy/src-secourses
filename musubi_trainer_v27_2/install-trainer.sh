
#!/bin/env bash

export UV_SKIP_WHEEL_FILENAME_CHECK=1
export UV_LINK_MODE=copy

export WORKSPACE_DIR=/workspace
export SECOURSES_MUSUBI_TRAINER_DIR=/workspace/SECourses_Musubi_Trainer
export CONVERT_TO_QUANT_DIR=${SECOURSES_MUSUBI_TRAINER_DIR}/convert_to_quant
export MUSUBI_TUNER_DIR=${SECOURSES_MUSUBI_TRAINER_DIR}/musubi-tuner

export LOGS=${SECOURSES_MUSUBI_TRAINER_DIR}/logs

mkdir -p ${LOGS}

export HF_HOME=${WORKSPACE_DIR}

#------------------------------------------------------------------------------
# Clone the SECourses_Musubi_Trainer repository
#------------------------------------------------------------------------------
cd ${WORKSPACE_DIR}
git clone --depth 1 https://github.com/FurkanGozukara/SECourses_Musubi_Trainer
cd ${SECOURSES_MUSUBI_TRAINER_DIR}
git reset --hard
git pull

#------------------------------------------------------------------------------
# Clone the convert_to_quant repository
#------------------------------------------------------------------------------
cd ${SECOURSES_MUSUBI_TRAINER_DIR}
git clone --depth 1 https://github.com/FurkanGozukara/convert_to_quant
cd ${CONVERT_TO_QUANT_DIR}
git reset --hard
git pull

#------------------------------------------------------------------------------
# Clone the musubi-tuner repository
#------------------------------------------------------------------------------
cd ${SECOURSES_MUSUBI_TRAINER_DIR}
git clone --depth 1 https://github.com/kohya-ss/musubi-tuner
cd ${MUSUBI_TUNER_DIR}
git reset --hard
git pull

#------------------------------------------------------------------------------
# Install base version of python and venv
#------------------------------------------------------------------------------
cd ${SECOURSES_MUSUBI_TRAINER_DIR}

# Check if Python 3.10 is available, if not install it
if ! command -v python3.10 &> /dev/null
then
    echo "Python 3.10 not found, installing..."
    apt-get update
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update
    apt-get install -y python3.10 python3.10-venv python3.10-dev
    echo "Python 3.10 installed successfully"
else
    echo "Python 3.10 is already installed"
fi

export TRAINER_VENV_DIR=${SECOURSES_MUSUBI_TRAINER_DIR}/venv

# Create venv with Python 3.10 if it doesn't exist
if [ ! -d "${TRAINER_VENV_DIR}" ]; then
    echo "Creating Python3.10-based virtual environment..."
    python3.10 -m venv "${TRAINER_VENV_DIR}"
else
    echo "Virtual environment already exists"
fi

# Activate the virtual environment
source ${TRAINER_VENV_DIR}/bin/activate

export PATH="$TRAINER_VENV_DIR/bin:$PATH"

export TRAINER_PYTHON=${TRAINER_VENV_DIR}/bin/python
export TRAINER_PIP=${TRAINER_VENV_DIR}/bin/pip
export TRAINER_UV=${TRAINER_VENV_DIR}/bin/uv

#------------------------------------------------------------------------------
# Basic Tooling
# Upgrade pip and install uv in the virtual environment
#------------------------------------------------------------------------------
cd ${SECOURSES_MUSUBI_TRAINER_DIR}
${TRAINER_PYTHON} -m pip install --upgrade pip
${TRAINER_PIP} install uv

# Install system requirements for musubi trainer
${TRAINER_UV} pip install -r ${WORKSPACE_DIR}/requirements_musubi_trainer.txt

# Install the musubi-tuner in editable mode
cd ${MUSUBI_TUNER_DIR}
${TRAINER_UV} pip install -e .

# Install convert_to_quant in editable mode
cd ${CONVERT_TO_QUANT_DIR}
${TRAINER_UV} pip install -e .

cd ${SECOURSES_MUSUBI_TRAINER_DIR}

echo "Installation completed. Check for errors."
