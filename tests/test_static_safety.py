"""Static smoke checks; no real hardware, no physical robot control."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_mapping_is_marked_unverified():
    d=json.loads((ROOT/'configs/mapping_candidate_UNVERIFIED.json').read_text('utf-8'))
    assert d['arm_signs']==[1,1,1,1,1]
    assert d['arm_offsets_rad']==[0,0,0,0,0]
    assert d['status'].startswith('UNVERIFIED')

def test_python_sources_parse():
    for f in (ROOT/'src').glob('*.py'):
        compile(f.read_text('utf-8'),str(f),'exec')
