"""Independent Poisson membership and one IID Gaussian release per logical step."""
import torch


def poisson_indices(population, q, generator):
    return (torch.rand(population,generator=generator)<q).nonzero().flatten()


@torch.no_grad()
def privatize(parameters, sums, sigma, generator):
    for p,total in zip(parameters,sums):
        noise=torch.randn(p.shape,dtype=p.dtype,device=p.device,generator=generator)
        p.grad=(total+sigma*30*noise)/1000
