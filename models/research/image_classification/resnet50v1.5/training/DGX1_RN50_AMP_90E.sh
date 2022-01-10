#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
WORKSPACE=${1:-"/media/resultsImagenetTF/"}
DATA_DIR=${2:-"/media/imageNetTF_20dir/"}

echo "WORKSPACE=$WORKSPACE"
echo "DATA_DIR=$DATA_DIR"

OTHER=${@:3}

if [[ ! -z "${BIND_TO_SOCKET}" ]]; then
    BIND_TO_SOCKET="--bind-to socket"
fi

python3 main.py --arch=resnet50 \
    --mode=train --iter_unit=epoch --num_iter=20 --use_rali \
    --batch_size=128 --warmup_steps=100 --use_cosine --label_smoothing 0.1 \
    --lr_init=0.256 --lr_warmup_epochs=8 --momentum=0.875 --weight_decay=3.0517578125e-05 \
    --use_tf_amp --use_static_loss_scaling --loss_scale 128 \
    --data_dir=${DATA_DIR}/tfr20 --data_idx_dir=${DATA_DIR}/dali_idx \
    --results_dir=${WORKSPACE}/results --weight_init=fan_in ${OTHER}

WORKSPACE=${1:-"/media/resultsImagenetTF/"}
DATA_DIR=${2:-"/media/imageNetTF_20dir/"}

echo "WORKSPACE=$WORKSPACE"
echo "DATA_DIR=$DATA_DIR"

OTHER=${@:3}

python3 main.py --arch=resnet50 \
    --mode=evaluate --iter_unit=epoch --num_iter=20 --use_rali \
    --batch_size=128 --warmup_steps=100 --use_cosine --label_smoothing 0.1 \
    --lr_init=0.256 --lr_warmup_epochs=8 --momentum=0.875 --weight_decay=3.0517578125e-05 \
    --use_tf_amp --use_static_loss_scaling --loss_scale 128 \
    --data_dir=${DATA_DIR}/tfr20 --data_idx_dir=${DATA_DIR}/dali_idx \
    --results_dir=${WORKSPACE}/results --weight_init=fan_in ${OTHER}
