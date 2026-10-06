from exp6.runtime import require_curve
require_curve()
import time,torch
from exp6.model import create_model
from exp6.clipping import per_example
from torch.nn.attention import sdpa_kernel, SDPBackend
torch.set_num_threads(2)
torch.manual_seed(20261001)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False
model,_=create_model(); model=model.cuda()
x=torch.randn(50,3,224,224,device='cuda'); y=torch.arange(50,device='cuda')%100
with sdpa_kernel(SDPBackend.MATH):
 for i in range(3):
  torch.cuda.synchronize(); start=time.monotonic()
  g,l,a=per_example(model,x,y)
  torch.cuda.synchronize(); print(i,time.monotonic()-start,torch.cuda.max_memory_allocated()/1024**3,flush=True)
  del g,l,a
