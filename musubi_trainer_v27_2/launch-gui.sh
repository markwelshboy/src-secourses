
#!/bin/env bash

export SECOURSES_MUSUBI_TRAINER_DIR=/workspace/SECourses_Musubi_Trainer
export TRAINER_VENV_DIR=${SECOURSES_MUSUBI_TRAINER_DIR}/venv
export LOGS=${SECOURSES_MUSUBI_TRAINER_DIR}/logs

mkdir -p ${LOGS}

# Activate the virtual environment
source ${TRAINER_VENV_DIR}/bin/activate

export PATH="$TRAINER_VENV_DIR/bin:$PATH"

export TRAINER_PYTHON=${TRAINER_VENV_DIR}/bin/python

unset LD_LIBRARY_PATH

cd ${SECOURSES_MUSUBI_TRAINER_DIR}

tmux new-session -d -s "musubi-trainer-gui" \
  "unset LD_LIBRARY_PATH && ${TRAINER_PYTHON} \"${SECOURSES_MUSUBI_TRAINER_DIR}/gui.py --share\" >> \"${LOGS}/gui.log\" 2>&1"
