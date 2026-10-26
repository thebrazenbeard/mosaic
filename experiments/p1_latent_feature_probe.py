"""Diagnostic: does the frozen source hidden state encode count information?"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import torch
from safetensors.torch import load_file, save_file
from p1_toy_latent_benchmark import episodes, features


def main():
    torch.manual_seed(20261026)
    torch.set_num_threads(4)
    root = Path(os.environ['MOSAIC_LATENT_ROOT']).resolve(strict=True)
    store = root / 'pretrained_bridge_pilot_1'
    hidden = load_file(str(store / 'source_hidden.safetensors'))['frozen_source_hidden']
    rows = episodes()
    training = rows['train'][:512]
    heldout = rows['run_a']+rows['run_b']
    _, query_train, label_train = features(training)
    _, query_test, label_test = features(heldout)
    x_train = torch.cat((hidden[:512],query_train),1)
    x_test = torch.cat((hidden[512:],query_test),1)
    head = torch.nn.Sequential(torch.nn.LayerNorm(x_train.shape[1]),
        torch.nn.Linear(x_train.shape[1],128),torch.nn.ReLU(),
        torch.nn.Linear(128,2))
    opt = torch.optim.AdamW(head.parameters(),lr=0.003)
    start=time.monotonic()
    for epoch in range(100):
        ix = torch.randperm(len(x_train))
        for b in ix.split(64):
            logits=head(x_train[b])
            loss=torch.nn.functional.cross_entropy(logits,label_train[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch in (0,9,29,59,99):
            with torch.inference_mode():
                pred=head(x_test).argmax(1)
                n=int((pred==label_test).sum())
            print('EPOCH',epoch+1,'LOSS',round(float(loss.detach()),5),
                  'HELDOUT_CORRECT',n,'/128','SEC',round(time.monotonic()-start,2),flush=True)
    with torch.inference_mode():
        preds=head(x_test).argmax(1).tolist()
    save_file({k:v.detach().contiguous() for k,v in head.state_dict().items()},
              str(store / 'feature_probe_head.safetensors'))
    (store / 'feature_probe_receipt.json').write_text(json.dumps({
        'correct':sum(a==b for a,b in zip(preds,label_test.tolist())),
        'total':128,'predictions':preds,'labels':label_test.tolist(),
        'scope':'posthoc_diagnostic_not_p1'},indent=2))


if __name__=='__main__':
    main()
