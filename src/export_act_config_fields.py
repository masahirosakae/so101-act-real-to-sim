#!/usr/bin/env python3
"""Extract *selected* ACT checkpoint config fields only. No inference/hardware."""
import argparse, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    cfg_file=a.checkpoint/'config.json'
    cfg=json.loads(cfg_file.read_text(encoding='utf-8-sig'))
    keys=('type','n_action_steps','chunk_size','use_vae','dim_model','n_encoder_layers','n_decoder_layers','vision_backbone','pretrained_backbone_weights','dropout','temporal_ensemble_coeff','optimizer_lr','optimizer_lr_backbone','optimizer_weight_decay','input_features','output_features','normalization_mapping')
    data={'source':'LOCAL saved checkpoint config.json','training_args_and_loss':'NOT_CONFIRMED: requires actual training logs','fields':{key:cfg.get(key,'NOT_PRESENT') for key in keys}}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    for key in ('type','n_action_steps','chunk_size','use_vae','vision_backbone'):
        print(key,repr(data['fields'][key]))
    print('Saved',a.output)
if __name__=='__main__':main()
