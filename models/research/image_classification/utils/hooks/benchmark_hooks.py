#! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import tensorflow as tf

import dllogger

from .training_hooks import MeanAccumulator


__all__ = ['BenchmarkLoggingHook']


class BenchmarkLoggingHook(tf.train.SessionRunHook):

    def __init__(self, global_batch_size, warmup_steps=20):
        self.latencies = []
        self.warmup_steps = warmup_steps
        self.global_batch_size = global_batch_size
        self.current_step = 0
        self.t0 = None
        self.mean_throughput = MeanAccumulator()
        # print("mean_throughput initail value:",self.mean_throughput)

    def before_run(self, run_context):
        self.t0 = time.time()
        # print("BEFORE BATCH TIME",self.t0)

    def after_run(self, run_context, run_values):
        batch_time = time.time() - self.t0
        ips = self.global_batch_size / batch_time
        # print("IPS ",ips)
        # print("AFTER BATCH TIME diff",batch_time)
        # print("current step:",self.current_step)
        # print("warmup steps",self.warmup_steps)
        if self.current_step >= self.warmup_steps:
            # print("CONDITION SATISFIED")
            self.latencies.append(batch_time)
            self.mean_throughput.consume(ips)
            # print("latenciessssssss:",self.latencies)
            # print("mean thorughputtttttt: ",self.mean_throughput)

            dllogger.log(data={"total_ips" : ips},
                         step=(0, self.current_step))

        self.current_step += 1
        print("current step *******************************",self.current_step)
