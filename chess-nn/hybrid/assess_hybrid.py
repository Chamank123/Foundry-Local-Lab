"""Validate final residual export plus combined engine score on fixed validation rows."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0,str(Path(__file__).resolve().parent/"engine"))
import chess
import search_numba as engine
from nn_adapter import score_nn
from nn_encode import encode_pychess
from nn_runtime import nn_forward_numpy
import nn_model
from residual_data import board_from_features
from check_export_parity import torch_model_from_npz

shards=Path(sys.argv[1])
checkpoint=sys.argv[2]
weights=nn_model.load_weights(checkpoint)
w=tuple(weights[k] for k in nn_model.WEIGHT_KEYS)
model=torch_model_from_npz(weights)
rows=[]
for path in sorted(shards.glob('shard_*.npz')):
    with np.load(path,allow_pickle=False) as z:
        keys=z['key']; selected=np.flatnonzero((keys%1000>=10)&(keys%1000<20))
        x,n,y=z['idx'],z['cnt'],z['y']
        for i in selected[:max(0,2000-len(rows))]:
            rows.append((x[i].copy(),int(n[i]),float(y[i])))
    if len(rows)>=2000:
        break
dense=np.zeros((len(rows),780),np.float32)
for i,(ids,n,_) in enumerate(rows):
    dense[i,ids[:n]]=1
with torch.no_grad():
    predicted=model(torch.from_numpy(dense)).numpy()
worst_float,worst_int=0.,0
base,total,labels=[],[],[]
for i,(ids,n,label) in enumerate(rows):
    bd=board_from_features(ids,n)
    neural=score_nn(bd,w)
    reference=nn_forward_numpy(ids,n,weights)
    hc=engine.evaluate_handcrafted(bd)
    actual=engine.evaluate(bd,w,4000)
    target_int=int(max(-6000,min(6000,hc+float(predicted[i]))))
    worst_float=max(worst_float,abs(neural-reference),abs(neural-float(predicted[i])))
    worst_int=max(worst_int,abs(actual-target_int))
    base.append(hc); total.append(hc+neural); labels.append(label)
assert worst_float<.01,("float mismatch",worst_float)
assert worst_int<=1,("combined integer mismatch",worst_int)
sigmoid=lambda x:1/(1+np.exp(-np.asarray(x)/400))
target=sigmoid(labels)
report={"rows":len(rows),"max_float_error_cp":worst_float,"max_combined_int_error_cp":worst_int,
        "baseline_validation_subset_rmse":float(np.sqrt(np.mean((sigmoid(base)-target)**2))),
        "residual_validation_subset_rmse":float(np.sqrt(np.mean((sigmoid(total)-target)**2)))}
# Independently validate all FEN audit examples from the received original source.
from train_nnue import _canonical_key,_mover_relative_cp
audit=json.loads((shards/'audit_sample.json').read_text())
for sample in audit:
    b=chess.Board(sample['fen'])
    ids=encode_pychess(b)
    assert sorted(ids)==sample['active']
    assert int(_canonical_key(ids))==sample['canonical_key']
    assert _mover_relative_cp(sample['selected_white_cp'],b.turn)==sample['mover_cp']
report['source_audit_examples_passed']=len(audit)
print(json.dumps(report,indent=2))
