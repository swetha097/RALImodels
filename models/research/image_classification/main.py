#!/usr/bin/env python
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

import os

import warnings
warnings.simplefilter("ignore")

from amd.rali.plugin.tf import RALIIterator
from amd.rali.pipeline import Pipeline
import amd.rali.ops as ops
import amd.rali.types as types

import tensorflow as tf

import horovod.tensorflow as hvd
import dllogger

from utils import hvd_utils
from utils import rali_utils
from runtime import Runner
from model.resnet import model_architectures

from utils.cmdline_helper import parse_cmdline

class HybridTrainPipe(Pipeline):
	def __init__(self, feature_key_map, tfrecordreader_type, batch_size, num_threads, device_id, data_dir, crop, rali_cpu = True):
		super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id,rali_cpu=rali_cpu)
		self.input = ops.TFRecordReader(path=data_dir, index_path = "", reader_type=tfrecordreader_type, user_feature_key_map=feature_key_map,
			features={
				'image/encoded':tf.FixedLenFeature((), tf.string, ""),
				'image/class/label':tf.FixedLenFeature([1], tf.int64,  -1),
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
		self.res = ops.Resize(device=rali_device, resize_x=crop[0], resize_y=crop[1])
		self.cmnp = ops.CropMirrorNormalize(device="cpu",
											output_dtype=types.FLOAT,
											output_layout=types.NCHW,
											crop=crop,
											image_type=types.RGB,
											mean=[0 ,0,0],
											std=[255,255,255])
		self.coin = ops.CoinFlip(probability=0.5)
		print('rali "{0}" variant'.format(rali_device))

	def define_graph(self):
		inputs = self.input(name ="Reader")
		images = inputs["image/encoded"]
		labels = inputs["image/class/label"]
		images = self.decode(images)
		images = self.res(images)
		rng = self.coin()
		output = self.cmnp(images, mirror = rng)
		return [output, labels]

class HybridValPipe(Pipeline):
	def __init__(self, feature_key_map, tfrecordreader_type, batch_size, num_threads, device_id, data_dir, scale, centreCrop, rali_cpu = True):
		super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id,rali_cpu=rali_cpu)
		self.input = ops.TFRecordReader(path=data_dir, index_path = "", reader_type=tfrecordreader_type, user_feature_key_map=feature_key_map,
			features={
				'image/encoded':tf.FixedLenFeature((), tf.string, ""),
				'image/class/label':tf.FixedLenFeature([1], tf.int64,  -1),
				'image/filename':tf.FixedLenFeature((), tf.string, "")
			}
		)
		rali_device = 'cpu' if rali_cpu else 'gpu'
		decoder_device = 'cpu' if rali_cpu else 'mixed'
		self.decode = ops.ImageDecoder(user_feature_key_map=feature_key_map, device=decoder_device, output_type=types.RGB)
		self.res = ops.Resize(device=rali_device, resize_x=scale[0], resize_y=scale[1])
		self.centrecrop = ops.CentreCrop(crop=centreCrop)
		self.cmnp = ops.CropMirrorNormalize(device="cpu",
											output_dtype=types.FLOAT,
											output_layout=types.NCHW,
											crop=centreCrop,
											image_type=types.RGB,
											mean=[0 ,0,0],
											std=[255,255,255])
		print('rali "{0}" variant'.format(rali_device))

	def define_graph(self):
		inputs = self.input(name ="Reader")
		images = inputs["image/encoded"]
		labels = inputs["image/class/label"]
		images = self.decode(images)
		images = self.res(images)
		images = self.centrecrop(images)
		output = self.cmnp(images)
		return [output, labels]

if __name__ == "__main__":

	nt = 1
	di = 0
	raliCPU = True
	trainCropSize = (224,224)
	valScaleSize = (256,256)
	valCentreCropSize = (224,224)
	TFRecordReaderType = 0
	featureKeyMap = {
		'image/encoded':'image/encoded',
		'image/class/label':'image/class/label',
		'image/filename':'image/filename'
	}
	
	tf.logging.set_verbosity(tf.logging.ERROR)

	FLAGS = parse_cmdline(model_architectures.keys())
	hvd.init()

	if hvd.rank() == 0:
		log_path = os.path.join(FLAGS.results_dir, FLAGS.log_filename)
		os.makedirs(FLAGS.results_dir, exist_ok=True)

		dllogger.init(
			backends=[
				dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE, filename=log_path),
				dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE)
			]
		)
	else:
		dllogger.init(backends=[])
	dllogger.log(data=vars(FLAGS), step='PARAMETER')

	runner = Runner(
		# ========= Model HParams ========= #
		n_classes=1001,
		architecture=FLAGS.arch,
		input_format='NHWC',
		compute_format=FLAGS.data_format,
		dtype=tf.float32 if FLAGS.precision == 'fp32' else tf.float16,
		n_channels=3,
		height=224,
		width=224,
		distort_colors=False,
		log_dir=FLAGS.results_dir,
		model_dir=FLAGS.model_dir if FLAGS.model_dir is not None else FLAGS.results_dir,
		data_dir=FLAGS.data_dir,
		data_idx_dir=FLAGS.data_idx_dir,
		weight_init=FLAGS.weight_init,
		use_xla=FLAGS.use_xla,
		use_tf_amp=FLAGS.use_tf_amp,
		use_rali=FLAGS.use_rali,
		gpu_memory_fraction=FLAGS.gpu_memory_fraction,
		gpu_id=FLAGS.gpu_id,
		seed=FLAGS.seed
	)

	trainRecordsPath = FLAGS.data_dir + "/train/"
	valRecordsPath = FLAGS.data_dir + "/val/"
	print("Train records path = " + trainRecordsPath)
	print("Val records path = " + valRecordsPath)

	if FLAGS.mode in ["train", "train_and_evaluate", "training_benchmark"]:
		train_imageIterator = 0
		if(FLAGS.use_rali):
			train_pipe = HybridTrainPipe(feature_key_map=featureKeyMap, tfrecordreader_type=TFRecordReaderType, batch_size=FLAGS.batch_size, num_threads=nt, device_id=di, data_dir=trainRecordsPath, crop=trainCropSize, rali_cpu=raliCPU) 
			train_pipe.build()
			train_imageIterator = RALIIterator(train_pipe)
			rali_utils.initialize_enumerator(train_imageIterator, 0)

		runner.train(
			train_imageIterator=train_imageIterator,
			iter_unit=FLAGS.iter_unit,
			num_iter=FLAGS.num_iter,
			run_iter=FLAGS.run_iter,
			batch_size=FLAGS.batch_size,
			warmup_steps=FLAGS.warmup_steps,
			log_every_n_steps=FLAGS.display_every,
			weight_decay=FLAGS.weight_decay,
			lr_init=FLAGS.lr_init,
			lr_warmup_epochs=FLAGS.lr_warmup_epochs,
			momentum=FLAGS.momentum,
			loss_scale=FLAGS.loss_scale,
			label_smoothing=FLAGS.label_smoothing,
			mixup=FLAGS.mixup,
			use_static_loss_scaling=FLAGS.use_static_loss_scaling,
			use_cosine_lr=FLAGS.use_cosine_lr,
			is_benchmark=FLAGS.mode == 'training_benchmark',
			use_final_conv=FLAGS.use_final_conv,
			quantize=FLAGS.quantize,
			symmetric=FLAGS.symmetric,
			quant_delay = FLAGS.quant_delay,
			use_qdq = FLAGS.use_qdq,
			finetune_checkpoint = FLAGS.finetune_checkpoint,
		)

	if FLAGS.mode in ["train_and_evaluate", 'evaluate', 'inference_benchmark']:

		if FLAGS.mode == 'inference_benchmark' and hvd_utils.is_using_hvd():
			raise NotImplementedError("Only single GPU inference is implemented.")

		elif not hvd_utils.is_using_hvd() or hvd.rank() == 0:
			val_imageIterator = 0
			if(FLAGS.use_rali):
				val_pipe = HybridValPipe(feature_key_map=featureKeyMap, tfrecordreader_type=TFRecordReaderType, batch_size=FLAGS.batch_size, num_threads=nt, device_id=di, data_dir=valRecordsPath, scale=valScaleSize, centreCrop=valCentreCropSize, rali_cpu=raliCPU) 
				val_pipe.build()
				val_imageIterator = RALIIterator(val_pipe)
				rali_utils.initialize_enumerator(val_imageIterator, 1)

			runner.evaluate(
				val_imageIterator=val_imageIterator,
				iter_unit=FLAGS.iter_unit if FLAGS.mode != "train_and_evaluate" else "epoch",
				num_iter=FLAGS.num_iter if FLAGS.mode != "train_and_evaluate" else 1,
				warmup_steps=FLAGS.warmup_steps,
				batch_size=FLAGS.batch_size,
				log_every_n_steps=FLAGS.display_every,
				is_benchmark=FLAGS.mode == 'inference_benchmark',
				export_dir=FLAGS.export_dir,
				quantize=FLAGS.quantize,
				symmetric=FLAGS.symmetric,
				use_final_conv=FLAGS.use_final_conv,
				use_qdq=FLAGS.use_qdq
			)

	if FLAGS.mode == 'predict':
		if FLAGS.to_predict is None:
			raise ValueError("No data to predict on.")

		if not os.path.isfile(FLAGS.to_predict):
			raise ValueError("Only prediction on single images is supported!")

		if hvd_utils.is_using_hvd():
			raise NotImplementedError("Only single GPU inference is implemented.")

		elif not hvd_utils.is_using_hvd() or hvd.rank() == 0:
			runner.predict(FLAGS.to_predict, quantize=FLAGS.quantize, symmetric=FLAGS.symmetric, use_qdq=FLAGS.use_qdq, use_final_conv=FLAGS.use_final_conv)
