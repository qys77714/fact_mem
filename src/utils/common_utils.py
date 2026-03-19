import argparse
import time
import numpy as np
import random
import logging
import os

def set_seed(seed):
	np.random.seed(seed)
	random.seed(seed)

def configure_logging(log_file_path, override=False):
	log_dir = os.path.dirname(log_file_path)
	if log_dir and not os.path.exists(log_dir):
		os.makedirs(log_dir)

	if os.path.exists(log_file_path) and override:
		os.remove(log_file_path)

	logger = logging.getLogger(log_file_path)
	logger.setLevel(logging.INFO)

	file_handler = logging.FileHandler(log_file_path)
	logger.addHandler(file_handler)

	return logger


class Timer:
	def __init__(self):
		self.start_time = None
		self.end_time = None

	def start(self):
		self.start_time = time.time()

	def end(self):
		"""return hours, minites and seconds"""
		if self.start_time is None:
			raise ValueError("The timer has not start, please use .start() first.")

		self.end_time = time.time()
		elapsed_time = self.end_time - self.start_time

		# 将秒数转换为时、分、秒
		hours = int(elapsed_time // 3600)
		minutes = int((elapsed_time % 3600) // 60)
		seconds = int(elapsed_time % 60)

		return hours, minutes, seconds

	# 示例用法
	# timer = Timer()
	# timer.start()
	# time.sleep(5)  # 模拟一些耗时操作
	# hours, minutes, seconds = timer.end()
	# print(f"The program has consumed {hours} h {minutes} m {seconds} s.")


__all__ = ["set_seed", "configure_logging", "Timer"]
