"""Exploratory frozen-LLM latent transfer pilot; NOT a P1 qualification.

Uses two pinned distinct local models, a frozen source hidden-state encoder,
and an actual trained continuous prefix projected into a frozen target LM.
Comparisons are exploratory; no heldout tuning is permitted after evaluation.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
import weakref

import torch
from safetensors.torch import save_file, load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from p1_toy_latent_benchmark import episodes, THRESHOLDS, features


SEED, STEPS, BATCH, LR = 20261026, 120, 4, 0.005


def prompt_state(row):
    flags = ' '.join(str((row['mask'] >> bit) & 1) for bit in range(16))
    return 'Observed sixteen binary flags: ' + flags + '.'


def question(row):
    return f'At least {THRESHOLDS[row["family"]]} flags equal one? Answer yes or no:'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def batch_ids(tokenizer, sentences):
    encoded = tokenizer(sentences, return_tensors='pt', padding=True)
    return encoded['input_ids'], encoded['attention_mask']


def classify(model, adapter, state, tok, sentences, options, grad=False):
    ids, mask = batch_ids(tok, sentences)
    embed = model.get_input_embeddings()(ids)
    if adapter is not None:
        prefix = adapter(state).unsqueeze(1)
        embed = torch.cat((prefix, embed), dim=1)
        mask = torch.cat((torch.ones(mask.size(0), 1, dtype=mask.dtype), mask), 1)
    output = model(inputs_embeds=embed, attention_mask=mask, use_cache=False)
    # Padding is on the right; the prediction follows the last actual token.
    positions = mask.sum(dim=1) - 1
    logits = output.logits[torch.arange(len(sentences)), positions.long()]
    return logits[:, options]


def main():
    start = time.monotonic()
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(4)
    root = Path(os.environ['MOSAIC_LATENT_ROOT']).resolve(strict=True)
    out_dir = root / 'pretrained_bridge_pilot_1'
    out_dir.mkdir(exist_ok=False)
    source_dir = root / 'models' / 'SmolLM2-135M-Instruct'
    target_dir = root / 'models' / 'SmolLM2-360M-Instruct'
    rows = episodes()
    # Fixed exploratory train budget. Existing run_a/run_b labels remain unseen.
    training = rows['train'][:512]
    heldout = rows['run_a'] + rows['run_b']
    all_rows = training + heldout
    tokenizer_a = AutoTokenizer.from_pretrained(source_dir, local_files_only=True,
                                               trust_remote_code=False)
    tokenizer_a.pad_token = tokenizer_a.eos_token
    source = AutoModelForCausalLM.from_pretrained(
        source_dir, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation='eager').eval()
    print('EXTRACTION_STARTED',len(all_rows),flush=True)
    hidden = []
    with torch.inference_mode():
        for offset in range(0, len(all_rows), 16):
            ids, mask = batch_ids(tokenizer_a,[prompt_state(row) for row in all_rows[offset:offset+16]])
            output = source(ids,attention_mask=mask,output_hidden_states=True)
            final = output.hidden_states[-1]
            pooled = (final*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
            hidden.append(pooled.cpu())
    hidden = torch.cat(hidden,dim=0)
    save_file({'frozen_source_hidden':hidden.contiguous()},str(out_dir/'source_hidden.safetensors'))
    ref = weakref.ref(source)
    del source,tokenizer_a,output,final,pooled,ids,mask
    gc.collect()
    if ref() is not None:
        raise RuntimeError('Source model retained before loading target')
    print('SOURCE_EVICTED',list(hidden.shape),round(time.monotonic()-start,3),flush=True)
    tokenizer_b = AutoTokenizer.from_pretrained(target_dir,local_files_only=True,
                                               trust_remote_code=False)
    tokenizer_b.pad_token = tokenizer_b.eos_token
    target = AutoModelForCausalLM.from_pretrained(
        target_dir,local_files_only=True,trust_remote_code=False,
        dtype=torch.float32,attn_implementation='eager').eval()
    target.requires_grad_(False)
    positive = tokenizer_b.encode(' yes',add_special_tokens=False)
    negative = tokenizer_b.encode(' no',add_special_tokens=False)
    if len(positive)!=1 or len(negative)!=1 or positive[0]==negative[0]:
        raise RuntimeError(f'Expected single-token choices, got {positive}/{negative}')
    options = [negative[0],positive[0]]
    adapter = torch.nn.Sequential(torch.nn.LayerNorm(hidden.shape[1]),
        torch.nn.Linear(hidden.shape[1],128),torch.nn.Tanh(),
        torch.nn.Linear(128,target.config.hidden_size))
    optimizer = torch.optim.AdamW(adapter.parameters(),lr=LR)
    train_labels = torch.tensor([row['label'] for row in training])
    questions = [question(row) for row in all_rows]
    print('BRIDGE_TRAIN_STARTED',STEPS,'tokens',options,'dims',list(hidden.shape),flush=True)
    for step in range(STEPS):
        indices = torch.randint(0,len(training),(BATCH,))
        scores = classify(target,adapter,hidden[indices],tokenizer_b,
                          [questions[i] for i in indices],options,grad=True)
        loss = torch.nn.functional.cross_entropy(scores,train_labels[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.0)
        optimizer.step()
        if step in (0,9,39,79,STEPS-1):
            print('STEP',step+1,'LOSS',float(loss.detach()),
                  'TIME',round(time.monotonic()-start,2),flush=True)
    save_file({k:v.detach().contiguous().cpu() for k,v in adapter.state_dict().items()},
              str(out_dir/'trained_bridge.safetensors'))
    score = {'trained':[],'full_history':[],'reset':[],'truth':[]}
    with torch.inference_mode():
        for offset in range(len(training),len(all_rows),4):
            subset = all_rows[offset:offset+4]
            qs = [question(row) for row in subset]
            state = hidden[offset:offset+len(subset)]
            scores = classify(target,adapter,state,tokenizer_b,qs,options)
            ret = classify(target,None,None,tokenizer_b,
                           [prompt_state(row)+' '+question(row) for row in subset],options)
            reset = classify(target,None,None,tokenizer_b,qs,options)
            score['trained']+=scores.argmax(1).tolist()
            score['full_history']+=ret.argmax(1).tolist()
            score['reset']+=reset.argmax(1).tolist()
            score['truth'] += [row['label'] for row in subset]
    for arm in ('trained','full_history','reset'):
        print('ARM',arm,'ACCURACY',sum(a==b for a,b in zip(score[arm],score['truth'])),
              '/',len(score['truth']),flush=True)
    outcome = {'scope':'exploratory_pretrained_bridge_not_p1',
               'adapter_sha256':digest(out_dir/'trained_bridge.safetensors'),
               'source_hidden_sha256':digest(out_dir/'source_hidden.safetensors'),
               'steps':STEPS,'batch':BATCH,'source_unloaded_before_target':True,
               'elapsed_seconds':round(time.monotonic()-start,3),'predictions':score}
    (out_dir/'receipt.json').write_text(json.dumps(outcome,indent=2),encoding='utf-8')
    print('PILOT_COMPLETE',json.dumps({k:v for k,v in outcome.items() if k!='predictions'}),flush=True)


if __name__=='__main__':
    main()
