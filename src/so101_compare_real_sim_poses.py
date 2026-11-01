#!/usr/bin/env python3
"""Render same-time recorded SO-101 joint poses as independent MuJoCo still images.

MuJoCo only: sets qpos and calls mj_forward; NO mj_step, ACT, robot motors,
COM ports, physical cameras, or other hardware.

This is VISUAL calibration assistance, not an automatic/verified calibration.
An overhead camera cannot unambiguously determine every joint's sign/offset.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def parse_args():
    p = argparse.ArgumentParser(description="Compare SAME-FRAME real image/state to MuJoCo static joint pose")
    p.add_argument("--reference-dir", type=Path, required=True, help="Directory created by exporter")
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--camera", default="car_goal_top", help="MJCF fixed camera used in scene_car_goal_real_layout.xml")
    p.add_argument("--sweep-joint", choices=JOINTS[:5], help="Generate +/- sign & offset candidate grid for one joint")
    p.add_argument("--sweep-frame", type=int, help="One of the exported frame indices; default middle frame")
    p.add_argument("--sweep-offset-step-deg", type=float, default=15.0,
                   help="Offset candidates relative to mapping: -2step,-step,0,+step,+2step")
    a = p.parse_args()
    if a.sweep_offset_step_deg <= 0 or a.sweep_offset_step_deg > 45:
        p.error("--sweep-offset-step-deg must be 0..45")
    return a


def get_map(path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("joint_order") != JOINTS:
        raise ValueError("mapping.joint_order must match SO101's 6 joint names")
    if cfg.get("arm_units") != "degrees":
        raise ValueError("This visual calibration tool requires real state values in DEGREES; check dataset")
    sign, off = cfg.get("arm_signs"), cfg.get("arm_offsets_rad")
    if not isinstance(sign, list) or len(sign) != 5 or any(float(x) not in (-1.0, 1.0) for x in sign):
        raise ValueError("Invalid arm_signs: expected 5 values of +/-1")
    if not isinstance(off, list) or len(off) != 5 or not all(math.isfinite(float(x)) for x in off):
        raise ValueError("Invalid arm_offsets_rad: expected 5 finite values")
    if cfg.get("gripper_mode") != "freeze":
        raise ValueError("Require fixed gripper; no gripper calibration from this dataset")
    return cfg


def mapping_pose(state, mapping, model, joint_ids):
    """Return converted joint positions with no silent clipping; float degrees input."""
    import numpy as np
    raw = np.array([math.radians(float(x)) * float(s) + float(o)
                    for x, s, o in zip(state[:5], mapping["arm_signs"], mapping["arm_offsets_rad"])])
    margin = np.asarray([model.jnt_range[j] for j in joint_ids[:5]], dtype=float)
    bad = [JOINTS[i] for i in range(5) if raw[i] < margin[i,0]+1e-4 or raw[i] > margin[i,1]-1e-4]
    return raw, bad


def render_state(model, renderer, camera, state, mapping, joint_ids, act_ids):
    import mujoco
    import numpy as np
    q, bad = mapping_pose(state, mapping, model, joint_ids)
    if bad:
        return None, bad, q
    d = mujoco.MjData(model)
    for k in range(5):
        d.qpos[model.jnt_qposadr[joint_ids[k]]] = q[k]
        d.ctrl[act_ids[k]] = q[k]
    grip = float(mapping.get("gripper_fixed_rad", 0))
    gl, gh = model.jnt_range[joint_ids[-1]]
    if not gl <= grip <= gh:
        raise ValueError("Mapping fixed gripper lies outside MuJoCo joint range")
    d.qpos[model.jnt_qposadr[joint_ids[-1]]] = grip
    d.ctrl[act_ids[-1]] = grip
    # IMPORTANT: static kinematic rendering. No physics stepping / control.
    mujoco.mj_forward(model, d)
    renderer.update_scene(d, camera=camera)
    return np.asarray(renderer.render()).copy(), [], q


def text_box(image, label, margin_top=35):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(image) if not isinstance(image, Image.Image) else image
    im = im.convert("RGB")
    canvas = Image.new("RGB", (im.width, im.height + margin_top), (25, 30, 36))
    canvas.paste(im, (0, margin_top))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), label, fill="white", font=ImageFont.load_default())
    return canvas


def pair(real, sim, ep, frame):
    from PIL import Image
    real = real.convert("RGB").resize((640,480))
    left = text_box(real, f"REAL | episode {ep} frame {frame} | recorded image & state")
    right = text_box(sim, "SIM | SAME episode frame | qpos SET ONLY (no dynamics)")
    out = Image.new("RGB", (1280,515), "white")
    out.paste(left,(0,0)); out.paste(right,(640,0))
    return out


def make_contact_sheet(pairs, path):
    from PIL import Image
    # Reduce heights so 7+ frames remain readable and reasonable in size.
    previews = [(k, im.resize((960,386))) for k,im in pairs]
    sheet = Image.new("RGB", (960, 386*len(previews)), (235,235,235))
    for row,(k,im) in enumerate(previews):
        sheet.paste(im,(0,386*row))
    sheet.save(path, quality=90)


def make_sweep(renderer, model, camera, sample, mapping, joint_ids, act_ids, real_image, args):
    from PIL import Image, ImageDraw, ImageFont
    name = args.sweep_joint
    idx = JOINTS.index(name)
    step = args.sweep_offset_step_deg
    signs = (-1,1)
    offsets_deg = (-2*step,-step,0,step,2*step)
    # 5 columns x 2 rows of simulations, recorded reference at top.
    cell_w, cell_h = 320, 270
    width, height = cell_w*5, 140+2*cell_h
    out = Image.new("RGB",(width,height),(238,241,245))
    ref = real_image.convert("RGB")
    ref.thumbnail((315,128))
    out.paste(ref,(8,8))
    d=ImageDraw.Draw(out)
    d.text((335,15),f"REFERENCE: EP {sample['episode_frame']} | joint={name}",fill=(0,0,0))
    d.text((335,42),"Candidates vary ONE joint's sign and offset; others use mapping file",fill=(0,0,0))
    d.text((335,69),"VISUAL CHECK ONLY - never auto-apply this to ACT or physical hardware",fill=(140,40,40))
    for row,sign in enumerate(signs):
        for col,offset in enumerate(offsets_deg):
            candidate = dict(mapping)
            candidate["arm_signs"] = list(mapping["arm_signs"])
            candidate["arm_offsets_rad"] = list(mapping["arm_offsets_rad"])
            candidate["arm_signs"][idx] = sign
            candidate["arm_offsets_rad"][idx] = float(mapping["arm_offsets_rad"][idx])+math.radians(offset)
            rgb,bad,q = render_state(model,renderer,camera,sample["state"],candidate,joint_ids,act_ids)
            x,y=col*cell_w,140+row*cell_h
            if rgb is None:
                tile=Image.new("RGB",(320,240),(210,208,207))
                ImageDraw.Draw(tile).text((8,110),"OUTSIDE JOINT LIMITS: "+','.join(bad),fill=(100,0,0))
            else:
                tile=Image.fromarray(rgb).resize((320,240))
            out.paste(tile,(x,y+30))
            d.text((x+8,y+8),f"{name}: sign {sign:+d}, offset change {offset:+g}deg",fill=(0,0,0))
    path = args.output / f"sweep_{name}_f{sample['episode_frame']:04d}.jpg"
    out.save(path,quality=89)
    print("Sweep candidate grid:",path)


def main():
    args = parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    import mujoco
    import numpy as np
    from PIL import Image
    info = json.loads((args.reference_dir/"real_pose_samples.json").read_text(encoding="utf-8"))
    if info.get("joint_order") != JOINTS or info.get("kind") != "recorded_so101_real_image_and_state":
        raise ValueError("Invalid reference JSON; run SO101 export tool first")
    samples=info.get("samples",[])
    if not samples:
        raise ValueError("No samples in reference JSON")
    mapping=get_map(args.mapping)
    model=mujoco.MjModel.from_xml_path(str(args.scene))
    if model.nq != 6 or model.nu != 6:
        raise ValueError("Expected 6 joint coordinates and 6 actuators (no moving car)")
    if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,args.camera)<0:
        raise ValueError(f"Camera not found in MJCF: {args.camera}")
    joint_ids=[];act_ids=[]
    for name in JOINTS:
        j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        a=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,name)
        if j<0 or a<0:
            raise ValueError(f"Scene missing joint/actuator {name}")
        if model.actuator_trnid[a,0]!=j:
            raise ValueError(f"Actuator does not control named joint {name}")
        joint_ids.append(j);act_ids.append(a)
    valid_samples=[];rows=[];images=[]
    renderer=mujoco.Renderer(model,height=480,width=640)
    try:
        for sample in sorted(samples,key=lambda s:int(s['episode_frame'])):
            f=int(sample['episode_frame'])
            real=Image.open(args.reference_dir/sample['image_file']).convert("RGB")
            state=np.asarray(sample['state'],dtype=float)
            if state.shape!=(6,) or not np.isfinite(state).all():
                raise ValueError(f"Invalid real joint state in frame {f}")
            sim,bad,q=render_state(model,renderer,args.camera,state,mapping,joint_ids,act_ids)
            row={"episode_frame":f,"dataset_index":sample["dataset_index"],"status":"SKIPPED_CLIPPED" if bad else "RENDERED",
                 "out_of_bounds_joints":';'.join(bad)}
            for i,name in enumerate(JOINTS[:5]):
                row[f"recorded_deg_{name}"]=float(state[i])
                row[f"sim_target_rad_{name}"]=float(q[i])
            rows.append(row)
            if bad:
                print(f"SKIPPED frame {f}: mapping outside model limits for {bad}. No silent clipping.")
                continue
            result=pair(real,sim,info['episode'],f)
            output=args.output/f"same_time_ep{info['episode']:02d}_frame{f:04d}.png"
            result.save(output)
            images.append((f,result));valid_samples.append(sample)
            print(f"Rendered REAL vs SIM frame {f}: {output}",flush=True)
        if images:
            contact=args.output/"same_time_pose_contact_sheet.jpg"
            make_contact_sheet(images,contact)
            print("Contact sheet:",contact)
        else:
            print("No frames rendered. Change sample frames or mapping; all current mappings exceeded model limits.")
        if args.sweep_joint:
            if args.sweep_frame is None:
                if not valid_samples:
                    raise ValueError("No in-range frames to use for sign/offset sweep")
                sample=valid_samples[len(valid_samples)//2]
            else:
                choices=[s for s in samples if int(s["episode_frame"])==args.sweep_frame]
                if not choices:
                    raise ValueError(f"No exported reference frame {args.sweep_frame}; choose from {sorted(int(s['episode_frame']) for s in samples)}")
                sample=choices[0]
            real=Image.open(args.reference_dir/sample['image_file']).convert('RGB')
            make_sweep(renderer,model,args.camera,sample,mapping,joint_ids,act_ids,real,args)
    finally:
        renderer.close()
    report=args.output/"same_time_pose_report.csv"
    with report.open("w",encoding="utf-8",newline="") as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0].keys()))
        writer.writeheader();writer.writerows(rows)
    (args.output/"mapping_used_UNVERIFIED.json").write_text(json.dumps(mapping,indent=2),encoding="utf-8")
    print("Report:", report)
    print(f"Rendered {len(valid_samples)} / {len(samples)} same-time frames")
    print("STATIC POSES ONLY: this DOES NOT verify joint signs, camera geometry, or task success.")
    print("NO ACT, NO mj_step, NO HARDWARE OR COM PORTS.")


if __name__=="__main__":
    main()
