
import os
import psutil

while True:
  pids = psutil.pids()
  if 17413 not in pids:
    break

os.system("CUDA_VISIBLE_DEVICES=0 python eval.py")
