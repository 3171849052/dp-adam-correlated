import time
from benchmark import test,OUT,read
while not (OUT/'capacity.json').exists():time.sleep(1)
n=read(OUT/'capacity.json')['max_success']
for g in ('adam','momentum'):
    if not test(n,g):raise RuntimeError(f'{g} failed at {n}')
assert test(12,'stpex2',1)
