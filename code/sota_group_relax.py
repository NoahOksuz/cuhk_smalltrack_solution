"""Compare clock-jitter-tolerant recording grouping against training identifiers."""
import argparse
import json
from collections import defaultdict
import numpy as np


def assign_relaxed(meta,gap=1.,slack=100.):
    groups=defaultdict(list)
    loose=[]
    for i,m in enumerate(meta):
        if 't0' in m and 'th0' in m:
            groups[(m['date'],tuple(m.get('devices',[])))].append(i)
        else:
            loose.append(i)
    result=[-1]*len(meta)
    block=0
    for ids in groups.values():
        ids.sort(key=lambda i:meta[i]['t0'])
        prev=None
        for i in ids:
            m=meta[i]
            same=False
            if prev is not None:
                p=meta[prev]
                dt=m['t0']-p['t0']; dh=m['th0']-p['th0']
                old=dh>0 and dt>=0 and abs(dh-24*dt)<max(30,.35*24*max(dt,1))
                adjacent=0<=dt<=60 and m['t0']-p.get('t1',p['t0'])<=gap
                jitter=abs(dh-24*dt)<max(slack,.75*24*dt)
                same=old or (adjacent and jitter)
            if not same:
                block+=1
            result[i]=block
            prev=i
    for i in loose:
        block+=1
        result[i]=block
    return result


def assign_adjacent(meta,gap=3.,slack=100.,loose_gap=3.):
    """assign_relaxed, then attach clips that have timestamps but no thermal counter
    to a block whose neighbouring clip ends or starts within loose_gap seconds.
    On training metadata this makes 128 links, none across (user, trial) takes."""
    blocks=assign_relaxed(meta,gap,slack)
    if loose_gap<=0:
        return blocks
    groups=defaultdict(list)
    for i,m in enumerate(meta):
        if 't0' in m:
            groups[(m['date'],tuple(m.get('devices',[])))].append(i)
    for ids in groups.values():
        ids.sort(key=lambda i:meta[i]['t0'])
        for p,i in enumerate(ids):
            if 'th0' in meta[i]:
                continue
            m=meta[i]
            if p>0:
                prev=meta[ids[p-1]]
                if 0<=m['t0']-prev.get('t1',prev['t0'])<=loose_gap:
                    blocks[i]=blocks[ids[p-1]]
                    continue
            if p+1<len(ids) and 'th0' in meta[ids[p+1]]:
                nxt=meta[ids[p+1]]
                if 0<=nxt['t0']-m.get('t1',m['t0'])<=loose_gap:
                    blocks[i]=blocks[ids[p+1]]
    return blocks


def main():
    from sota_data import OUT,manifests
    from sota_blocks import scenes
    from sota_finish2 import structure
    from sota_structure import average_repetitions
    from sota_context_prior import fit_prior,decode
    ap=argparse.ArgumentParser()
    ap.add_argument('--confirm',action='store_true')
    ap.add_argument('--bits',type=int,default=5,choices=[5,6])
    args=ap.parse_args()
    rows,_=manifests()
    y=np.array([int(r['action_id']) for r in rows])
    meta=json.loads((OUT/'blocks.json').read_text())
    truth=[(r['user'],r['trial'],meta['train'][i].get('date')) for i,r in enumerate(rows)]
    fold=np.full(len(y),-1)
    ids=[]
    for f in range(5):
        v=np.load(OUT/f'scn{f}_reg_val.npz')['ids']; ids.append(v);fold[v]=f
    raw=(np.load(OUT/f'oof_reg_b{args.bits}.npy')+np.load(OUT/'oof_thmreg_b5.npy'))/2
    for gap,slack in ([(3,100)] if args.confirm else [(1,60),(1,100),(1,200),(3,100),(3,200),(6,200)]):
        m=meta['train']
        blocks=assign_relaxed(m,gap,slack)
        scene,members=scenes(m,blocks)
        impure=sum(len(set(truth[i] for i in r))>1 for r in members.values())
        repeats=sum(len(set(y[r]))<len(r) for r in members.values())
        cross=sum(len(set(fold[r]))>1 for r in members.values())
        print('GROUP',gap,slack,'blocks',len(members),'impure',impure,'repeated labels',repeats,'crossfold',cross,flush=True)
        if impure or repeats:
            continue
        assert cross==0, 'New recording group crosses existing model folds'
        predictions=np.zeros(len(y),int)
        scores=[]
        for f in (range(5) if args.confirm else range(3)):
            va=ids[f];tr=np.setdiff1d(np.arange(len(y)),va)
            prior=fit_prior(y,members,scene,tr)
            vm,vs=structure(m,blocks,va.tolist())
            guess=decode(average_repetitions(raw[va],vm,vs,.7),vm,prior,0.,.3)
            predictions[va]=guess
            scores.append(float((guess==y[va]).mean()))
            print(' accuracy',f,scores[-1],flush=True)
        if args.confirm:
            base=np.load(OUT/'context_prior_reg_pred.npy')
            hold=np.concatenate(ids[3:])
            delta=(predictions==y).astype(float)-(base==y).astype(float)
            from sota_blocks import regions
            groups=np.array(regions(meta['train'],meta['train_blocks'])[0])
            sums=np.array([delta[groups==g].sum() for g in np.unique(groups)])
            sizes=np.array([(groups==g).sum() for g in np.unique(groups)])
            draws=np.random.default_rng(42).integers(len(sizes),size=(5000,len(sizes)))
            boot=sums[draws].sum(1)/sizes[draws].sum(1)
            result=dict(gap=gap,slack=slack,visual_bits=args.bits,accuracy=float((predictions==y).mean()),baseline=float((base==y).mean()),
                        confirmation=float((predictions[hold]==y[hold]).mean()),confirmation_base=float((base[hold]==y[hold]).mean()),
                        folds=scores,delta_ci95=np.quantile(boot,[.025,.975]).tolist(),crossfold_blocks=cross,impure_blocks=impure)
            suffix='' if args.bits==5 else f'_b{args.bits}'
            (OUT/f'group_relax_confirmed{suffix}.json').write_text(json.dumps(result,indent=2))
            np.save(OUT/f'group_relax_pred{suffix}.npy',predictions)
            print('CONFIRMED',result,flush=True)


if __name__=='__main__':
    main()
