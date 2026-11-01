#!/usr/bin/env python3
"""CSV-only SO-101 timing diagnostic; no inference, robot, COM, or MuJoCo calls.

Reads predictions.csv from so101_act_offline_eval.py (stride=1) and compares:
- ACT prediction[t] vs teacher action[t+lag]
- recorded teacher action[t] vs *measured* state[t+lag]
- ACT vs a causal constant-velocity baseline for the +9 frame forecast

No reinterpretation of action[t] as desired physical pose: recorded teacher actions
may differ from real motor positions due to servo latency and/or recording semantics.
"""
import argparse
import csv
import math
from pathlib import Path

JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def corr(a, b):
    if len(a) < 3 or len(a) != len(b):
        return float('nan')
    ma, mb = mean(a), mean(b)
    aa = [x-ma for x in a]
    bb = [y-mb for y in b]
    den = math.sqrt(sum(x*x for x in aa) * sum(y*y for y in bb))
    return sum(x*y for x,y in zip(aa,bb))/den if den > 1e-12 else float('nan')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--max-lag', type=int, default=15)
    p.add_argument('--forecast-lag', type=int, default=9)
    p.add_argument('--fps', type=float, default=15)
    args=p.parse_args()
    if args.max_lag < 1 or args.fps <= 0 or not -args.max_lag <= args.forecast_lag <= args.max_lag:
        p.error('Bad --max-lag, --forecast-lag, or --fps')
    with args.csv.open(encoding='utf-8-sig', newline='') as f:
        data = list(csv.DictReader(f))
    needed={'episode_frame'} | {f'{field}_{j}' for field in ('state','teacher','pred') for j in JOINTS}
    if not data or needed - set(data[0]):
        p.error('Missing columns: '+', '.join(sorted(needed - set(data[0]) if data else needed)))
    rows={int(r['episode_frame']):r for r in data}
    if len(rows)!=len(data): p.error('Duplicate episode_frame values')
    # all lags and 5 past observations available for each reference frame
    common=[t for t in sorted(rows) if all((t+k) in rows for k in range(-args.max_lag, args.max_lag+1)) and (t-5) in rows]
    if len(common)<50: p.error('Not enough common frames: use stride=1 450-frame predictions.csv')
    def v(t, field, j):return float(rows[t][f'{field}_{j}'])
    def m_loss(field_a, field_b, lag, joint=None):
        js=(joint,) if joint else JOINTS
        return mean([abs(v(t,field_a,j)-v(t+lag,field_b,j)) for t in common for j in js])
    def deriv_c(field_a, field_b, lag, joint):
        aa=[v(t,field_a,joint)-v(t-1,field_a,joint) for t in common]
        bb=[v(t+lag,field_b,joint)-v(t+lag-1,field_b,joint) for t in common]
        return corr(aa,bb)
    lags=range(-args.max_lag,args.max_lag+1)
    lag=args.forecast_lag
    print('SO-101 action/state timing audit. CSV ONLY: no inference/hardware/MuJoCo.')
    print(f'Common frames={len(common)}, fps={args.fps:g}; lag convention: A[t] vs B[t+lag].')
    print('\nMODEL CHECK -- ACT[t] vs TEACHER[t+lag]:')
    modelbest=min(lags,key=lambda k:m_loss('pred','teacher',k))
    print(f'  best mean MAE lag={modelbest:+d} ({modelbest/args.fps:+.3f}s)')
    print(f'  lag 0: {m_loss("pred","teacher",0):.3f} deg; lag {lag:+d}: {m_loss("pred","teacher",lag):.3f} deg')
    print('\nRECORDING / SERVO FOLLOWING -- TEACHER ACTION[t] vs MEASURED STATE[t+lag]:')
    all_best=min(lags,key=lambda k:m_loss('teacher','state',k))
    print(f'  five-axis best lag={all_best:+d} ({all_best/args.fps:+.3f}s)')
    print(f'{"joint":17s} {"lag":>5} {"mae@0":>10} {"mae@best":>10} {"dCorr@0":>10} {"dCorr@best":>12}')
    for j in JOINTS:
        best=min(lags,key=lambda k:m_loss('teacher','state',k,j))
        print(f'{j:17s} {best:+5d} {m_loss("teacher","state",0,j):10.3f} {m_loss("teacher","state",best,j):10.3f} {deriv_c("teacher","state",0,j):10.3f} {deriv_c("teacher","state",best,j):12.3f}')
    print('\nFORECAST BASELINES -- all compare TEACHER[t+lag] on SAME timestamps:')
    act=m_loss('pred','teacher',lag)
    hold=m_loss('state','teacher',lag)
    # Zero-order hold (state_t) and causal linear extrapolation using past 5 recorded positions.
    extrap=mean([abs((v(t,'state',j)+lag*(v(t,'state',j)-v(t-5,'state',j))/5)-v(t+lag,'teacher',j)) for t in common for j in JOINTS])
    print(f'  lag {lag:+d}: ACT={act:.3f} deg; hold={hold:.3f} deg; causal 5-frame velocity extrapolation={extrap:.3f} deg')
    print('  extrapolation uses state[t] and state[t-5] ONLY, no future inputs.')
    print('\nCAUTION: alignment of predicted absolute joint positions with future trajectories')
    print('can improve even without a recording synchronization bug. Derivative correlation')
    print('and servo tracking results are supporting diagnostics, not proof.')

if __name__=='__main__':main()
