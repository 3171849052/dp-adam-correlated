"""Keep inherited flock alive throughout the persistent pipeline."""
import argparse,fcntl,os
p=argparse.ArgumentParser();p.add_argument('--lock-fd',type=int,required=True);args=p.parse_args()
# Never release this descriptor until the scheduler exits.
os.fstat(args.lock_fd);fcntl.flock(args.lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
from exp16a.pipeline import run
run()
