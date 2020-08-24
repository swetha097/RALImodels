# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
"""Binary to run train and evaluation on object detection model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from amd.rali.plugin.tf import RALIIterator
from amd.rali.pipeline import Pipeline
import amd.rali.ops as ops
import amd.rali.types as types

from absl import flags

import tensorflow as tf
import horovod.tensorflow as hvd
import dllogger
import time
import os
# tf.enable_eager_execution()
# import tensorflow.contrib.eager as tfe

from object_detection import model_hparams
from object_detection import model_lib
from object_detection.utils.exp_utils import AverageMeter, setup_dllogger

flags.DEFINE_string(
		'model_dir', None, 'Path to output model directory '
		'where event and checkpoint files will be written.')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
										'file.')
flags.DEFINE_string("raport_file", default="summary.json",
												 help="Path to dlloger json")
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_boolean('eval_training_data', False,
										 'If training data should be evaluated for this job. Note '
										 'that one call only use this in eval-only mode, and '
										 '`checkpoint_dir` must be supplied.')
flags.DEFINE_integer('sample_1_of_n_eval_examples', 1, 'Will sample one of '
										 'every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
										 'one of every n train input examples for evaluation, '
										 'where n is provided. This is only used if '
										 '`eval_training_data` is True.')
flags.DEFINE_integer('eval_count', 1, 'How many times the evaluation should be run')
flags.DEFINE_string(
		'hparams_overrides', None, 'Hyperparameter overrides, '
		'represented as a string containing comma-separated '
		'hparam_name=value pairs.')
flags.DEFINE_string(
		'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
		'`checkpoint_dir` is provided, this binary operates in eval-only mode, '
		'writing resulting metrics to `model_dir`.')
flags.DEFINE_boolean(
		'allow_xla', False, 'Enable XLA compilation')
flags.DEFINE_boolean(
		'amp', False, 'Whether to enable AMP ops. When false, uses TF32 on A100 and FP32 on V100 GPUS.')
flags.DEFINE_boolean(
		'run_once', False, 'If running in eval-only mode, whether to run just '
		'one round of eval vs running continuously (default).'
)
FLAGS = flags.FLAGS

class DLLoggerHook(tf.estimator.SessionRunHook):
	def __init__(self, global_batch_size, rank=-1):
		self.global_batch_size = global_batch_size
		self.rank = rank
		setup_dllogger(enabled=True, filename=FLAGS.raport_file, rank=rank)

	def after_create_session(self, session, coord):
		self.meters = {}
		warmup = 100
		self.meters['train_throughput'] = AverageMeter(warmup=warmup)

	def before_run(self, run_context):
		self.t0 = time.time()
		return tf.estimator.SessionRunArgs(fetches=['global_step:0', 'learning_rate:0'])

	def after_run(self, run_context, run_values):
		throughput = self.global_batch_size/(time.time() - self.t0)
		global_step, lr = run_values.results
		self.meters['train_throughput'].update(throughput)

	def end(self, session):
		summary = {
			'train_throughput': self.meters['train_throughput'].avg,
		}
		dllogger.log(step=tuple(), data=summary)

class HybridPipe(Pipeline):
	def __init__(self, feature_key_map, tfrecordreader_type, batch_size, num_threads, device_id, data_dir, crop, rali_cpu = True):
		super(HybridPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id,rali_cpu=rali_cpu)
		self.input = ops.TFRecordReader(path=data_dir, index_path = "", reader_type=tfrecordreader_type, user_feature_key_map=feature_key_map, 
			features={
				'image/encoded':tf.FixedLenFeature((), tf.string, ""),
				'image/class/label':tf.FixedLenFeature([1], tf.int64,  -1),
				'image/class/text':tf.FixedLenFeature([ ], tf.string, ''),
				'image/object/bbox/xmin':tf.VarLenFeature(dtype=tf.float32),
				'image/object/bbox/ymin':tf.VarLenFeature(dtype=tf.float32),
				'image/object/bbox/xmax':tf.VarLenFeature(dtype=tf.float32),
				'image/object/bbox/ymax':tf.VarLenFeature(dtype=tf.float32),
				'image/filename':tf.FixedLenFeature((), tf.string, "")
			}
		)
		rali_device = 'cpu' if rali_cpu else 'gpu'
		decoder_device = 'cpu' if rali_cpu else 'mixed'
		device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
		host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
		self.decode = ops.ImageDecoderRandomCrop(user_feature_key_map=feature_key_map,
												device=decoder_device, output_type=types.RGB,
												device_memory_padding=device_memory_padding,
												host_memory_padding=host_memory_padding,
												random_aspect_ratio=[0.8, 1.25],
												random_area=[0.1, 1.0],
												num_attempts=100)
		self.res = ops.Resize(device=rali_device, resize_x=crop, resize_y=crop)
		self.cmnp = ops.CropMirrorNormalize(device="cpu",
											output_dtype=types.FLOAT,
											output_layout=types.NCHW,
											crop=(crop, crop),
											image_type=types.RGB,
											mean=[0.485 * 255,0.456 * 255,0.406 * 255],
											std=[0.229 * 255,0.224 * 255,0.225 * 255])
		self.coin = ops.CoinFlip(probability=0.5)
		print('rali "{0}" variant'.format(rali_device))

	def define_graph(self):
		inputs = self.input(name ="Reader")
		images = inputs["image/encoded"]
		labels = inputs["image/class/label"]
		images = self.decode(images)
		images = self.res(images)
		rng = self.coin()
		output = self.cmnp(images, mirror=rng)
		return [output, labels]


def main(unused_argv):
	
	imagePath = "/media/ssdTraining/coco_1tfr/"
	bs = 2
	nt = 1
	di = 0
	raliCPU = True
	cropSize = 320
	TFRecordReaderType = 1
	featureKeyMap = {
		'image/encoded':'image/encoded',
		'image/class/label':'image/object/class/label',
		'image/class/text':'image/object/class/text',
		'image/object/bbox/xmin':'image/object/bbox/xmin',
		'image/object/bbox/ymin':'image/object/bbox/ymin',
		'image/object/bbox/xmax':'image/object/bbox/xmax',
		'image/object/bbox/ymax':'image/object/bbox/ymax',
		'image/filename':'image/filename'
	}

	pipe = HybridPipe(feature_key_map=featureKeyMap, tfrecordreader_type=TFRecordReaderType, batch_size=bs, num_threads=nt, device_id=di, data_dir=imagePath, crop=cropSize, rali_cpu=raliCPU) 
	pipe.build()
	
	imageIterator =  RALIIterator(pipe)	
	
	tf.logging.set_verbosity(tf.logging.INFO)
	if FLAGS.amp:
			os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
	else:
			os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "0"

	hvd.init()

	flags.mark_flag_as_required('model_dir')
	flags.mark_flag_as_required('pipeline_config_path')
	session_config = tf.ConfigProto()
	session_config.gpu_options.per_process_gpu_memory_fraction=0.9
	session_config.gpu_options.visible_device_list = str(hvd.local_rank())
	if FLAGS.allow_xla:
			session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
	model_dir = FLAGS.model_dir if hvd.rank() == 0 else None
	config = tf.estimator.RunConfig(model_dir=model_dir, session_config=session_config)

	train_and_eval_dict = model_lib.create_estimator_and_inputs(
		iterator=imageIterator,
		run_config=config,
		eval_count=FLAGS.eval_count,
		hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
		pipeline_config_path=FLAGS.pipeline_config_path,
		train_steps=FLAGS.num_train_steps,
		sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
		sample_1_of_n_eval_on_train_examples=(
				FLAGS.sample_1_of_n_eval_on_train_examples))
	estimator = train_and_eval_dict['estimator']
	train_input_fn = train_and_eval_dict['train_input_fn']
	eval_input_fns = train_and_eval_dict['eval_input_fns']
	eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
	predict_input_fn = train_and_eval_dict['predict_input_fn']
	train_steps = train_and_eval_dict['train_steps']

	if FLAGS.checkpoint_dir:
		if FLAGS.eval_training_data:
			name = 'training_data'
			input_fn = eval_on_train_input_fn
		else:
			name = 'validation_data'
			# The first eval input will be evaluated.
			input_fn = eval_input_fns[0]
		if FLAGS.run_once:
			estimator.evaluate(input_fn,
							steps=None,
							checkpoint_path=tf.train.latest_checkpoint(
									FLAGS.checkpoint_dir))
		else:
			model_lib.continuous_eval(estimator, FLAGS.checkpoint_dir, input_fn,
																train_steps, name)
	else:
		train_spec, eval_specs = model_lib.create_train_and_eval_specs(
				train_input_fn,
				eval_input_fns,
				eval_on_train_input_fn,
				predict_input_fn,
				train_steps,
				eval_on_train_data=False)

		train_hooks = [hvd.BroadcastGlobalVariablesHook(0), DLLoggerHook(hvd.size()*train_and_eval_dict['train_batch_size'], hvd.rank())]
		eval_hooks = []

		for x in range(FLAGS.eval_count):
			estimator.train(train_input_fn,
							hooks=train_hooks,
							steps=train_steps // FLAGS.eval_count)
			if hvd.rank() == 0:
				eval_input_fn = eval_input_fns[0]
				results = estimator.evaluate(eval_input_fn,
											steps=None,
											hooks=eval_hooks)

if __name__ == '__main__':
	tf.app.run()
