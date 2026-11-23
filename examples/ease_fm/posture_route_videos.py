"""Outcome-labeled, trace-verified scientific replays from saved E173 states."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json,shutil
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from .posture_route import ROOT
MODES=("frozen_bias",)
from .benchmark import fingerprint

def run():
    rows=json.loads((ROOT/'video_records.json').read_text());out=ROOT/'videos';out.mkdir(exist_ok=False)
    selection=[(r['case'],r['method'],r['mode']) for r in rows]
    manifest=dict(selection='PRIVILEGED route execution, original vs extended descent budget. Matched E170 and E120+release; fitting-only successes and failures.',playback='Saved executed states, every second40ms control step at12.5fps (1x).',videos=[])
    for case,method,mode in selection:
        stem=f'{case}_{method}_{mode}';r=json.loads((ROOT/'trials'/f'{stem}.json').read_text());label='success' if r['success'] else 'failure'
        m=mujoco.MjModel.from_xml_path(str((ROOT/'states'/f'{stem}.xml').resolve()));d=mujoco.MjData(m)
        states=np.load(ROOT/'states'/f'{stem}.npz')['states'];assert len(states)==len(r['trace'])
        site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_center');obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'object')
        renderer=mujoco.Renderer(m,height=480,width=640);camera=mujoco.MjvCamera();camera.lookat[:]=[.25,0,.30];camera.distance=2.1;camera.azimuth=130;camera.elevation=-25
        file=f'{stem}_{label}.mp4';writer=imageio.get_writer(out/file,fps=12.5,codec='libx264',quality=8,macro_block_size=16);frames=0
        for i,state in enumerate(states):
            d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];mujoco.mj_forward(m,d)
            np.testing.assert_allclose(d.site_xpos[site],r['trace'][i]['position'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(d.xpos[obj],r['trace'][i]['object'],atol=1e-12,rtol=0)
            if i%2:continue
            renderer.update_scene(d,camera=camera);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,640,62),fill='black');draw.text((10,6),f'E173 | {r["dof"]} joints | case {case} | {method} / {mode}',fill='white')
            draw.text((10,24),f'{label.upper()} | {r["trace"][i]["phase"]} | {(i+1)*.04:.2f}s',fill='#9cde19')
            draw.text((10,42),'PRIVILEGED geometry constraint diagnostic' if method.startswith('privileged_') else 'BASELINE E120 + release | No geometry constraint',fill='white')
            writer.append_data(np.asarray(im));frames+=1
            if i==100:im.save(out/f'{stem}_preview.png')
        writer.close();renderer.close();reader=imageio.get_reader(out/file);assert reader.count_frames()==frames;reader.close()
        manifest['videos'].append(dict(file=file,case=case,method=method,mode=mode,success=r['success'],collision=r['collision'],frames=frames,all_saved_states_match_trace=True,sha256=fingerprint(out/file)))
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(file,flush=True)
    html='<html><body style="background:#111;color:#eee;font:16px sans-serif;margin:30px"><h1>E173 / PRIVILEGED posture/route comparison</h1><p>'+manifest['selection']+'</p><p>'+manifest['playback']+'</p><p><a style="color:#9cde19" href="../report.md">Technical report</a></p><img width="1000" src="../comparison.png">'
    for v in manifest['videos']:html+=f'<h2>{v["file"]}</h2><video controls preload="none" width="640" src="{v["file"]}"></video>'
    (out/'index.html').write_text(html+'</body></html>');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    (ROOT/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'},indent=2))

if __name__=='__main__':run()
