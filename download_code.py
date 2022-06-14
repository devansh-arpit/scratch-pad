import torch_xla.distributed.xla_multiprocessing as xmp
import os

def _mp_fn(rank, flags):
	os.system('gsutil cp -r gs://sfr-tpu-us-east1-research/darpit/code /home/darpit/')
if __name__ == '__main__':
	print('downloading code...')
	xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork') # 