"""Record the real clipped aggregate before the private noise release."""
import hashlib
import torch
from exp2.scale import LogicalBatch as AdamLogicalBatch
from exp3.mechanism import LogicalBatch as MuonLogicalBatch
from exp3b.spec import output_path, write_json
from exp3b.support import expected_names


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class SignalWriter:
    def __init__(self, directory, selected, method, metadata):
        self.directory = output_path(directory)
        self.directory.mkdir(parents=True)
        assert tuple(selected) == expected_names()
        self.selected, self.method, self.metadata = selected, method, metadata
        self.files = []

    @torch.no_grad()
    def record(self, step, parameters, totals, logical_size, scaled=False):
        assert step == len(self.files) + 1
        sums = {id(p): total for p, total in zip(parameters, totals)}
        tensors = {}
        for name, p in self.selected.items():
            total = sums[id(p)]
            if scaled:
                # q is in average-gradient units; sigma_sum / B is used in replay.
                scale = p._logical_scale
                q = (total * scale) / logical_size
                g = q / scale
                tensors[name] = dict(g=g.cpu().clone(), q=q.cpu().clone(),
                                     scale=scale.cpu().clone())
            else:
                tensors[name] = dict(g=(total / logical_size).cpu().clone())
            assert all(torch.isfinite(value).all() for value in tensors[name].values())
        file = self.directory / f'step_{step:03d}.pt'
        torch.save(dict(step=step, tensors=tensors), file)
        self.files.append(dict(file=file.name, sha256=file_digest(file)))

    def finish(self, steps):
        assert steps == len(self.files)
        write_json(self.directory / 'manifest.json', dict(
            status='completed', method=self.method, steps=steps,
            support=list(self.selected), shapes={n: list(p.shape) for n, p in self.selected.items()},
            dimension=sum(p.numel() for p in self.selected.values()),
            coordinates='post-clipping pre-noise average gradient',
            scale_signal='g = q / S; q = S * clipped_sum / logical_batch_size',
            files=self.files, **self.metadata))


class CapturedAdamBatch(AdamLogicalBatch):
    def __init__(self, *args, writer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = writer

    @torch.no_grad()
    def finish_microbatch(self):
        # Snapshot the accumulated current gradients BEFORE the base implementation
        # transforms its sums in place and consumes its single private release.
        if self.writer is not None and self.micro_steps + 1 == self.accumulation:
            totals = [total + p.grad.float() for total, p in zip(self.sums, self.parameters)]
            self.writer.record(self.optimizer_steps + 1, self.parameters, totals,
                               self.logical_size, scaled=self.scaled)
        return super().finish_microbatch()


class CapturedMuonBatch(MuonLogicalBatch):
    def __init__(self, *args, writer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = writer

    @torch.no_grad()
    def finish_microbatch(self):
        if self.writer is not None and self.micro_steps + 1 == self.accumulation:
            assert self.method == 'mf_muon_standard'
            totals = [total + p.grad for total, p in zip(self.sums, self.parameters)]
            self.writer.record(self.optimizer_steps + 1, self.parameters, totals, self.logical_size)
        return super().finish_microbatch()
