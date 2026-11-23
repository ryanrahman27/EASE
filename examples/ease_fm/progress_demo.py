"""Captioned research reel: original motion graphics plus verified simulator footage."""
import json
import math
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont,ImageOps
from .benchmark import fingerprint

ROOT=Path('outputs/ease_progress_demo_v1')
W,H,FPS=1280,720,24
BG=(10,16,29);WHITE=(235,243,252);MUTED=(153,174,197);CYAN=(75,226,220);AMBER=(255,192,100)
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
def font(n,bold=False):return ImageFont.truetype(BOLD if bold else FONT,n)
def text(d,xy,s,size=24,color=WHITE,bold=False):d.text(xy,s,font=font(size,bold),fill=color)
def box(d,xy,outline=(42,64,86),fill=(17,28,44)):d.rounded_rectangle(xy,radius=18,fill=fill,outline=outline,width=2)

def run():
    ROOT.mkdir(exist_ok=True)
    paths={
      'train_old':Path('outputs/ease_progress_selector_v1/videos/case1_seed0_None_baseline_failure.mp4'),
      'train_new':Path('outputs/ease_progress_selector_v1/videos/case1_seed0_None_physics_success.mp4'),
      'transfer_old':Path('outputs/ease_unseen_progress_v1/videos/case10_seed0_None_baseline_failure.mp4'),
      'transfer_new':Path('outputs/ease_unseen_progress_v1/videos/case10_seed0_None_physics_success.mp4'),
      'failure':Path('outputs/ease_unseen_progress_v1/videos/case0_seed2_None_physics_failure.mp4')}
    clips={};sources=[]
    for key,p in paths.items():
        manifest=json.loads((p.parent/'manifest.json').read_text());v=next(v for v in manifest['videos'] if v['file']==p.name);assert v['exact_replay'] and fingerprint(p)==v['sha256']
        reader=imageio.get_reader(p);clips[key]=[Image.fromarray(a) for a in reader];reader.close();sources.append(dict(key=key,path=str(p),sha256=v['sha256'],verified_replay=True))
    latest=json.loads(Path('outputs/ease_release_selector_v1/summary.json').read_text());assert [r['success'] for r in latest['table'] if r['group']=='all']==[14,16]
    scenes=[(0,7,'The goal'),(7,17,'Read the history'),(17,29,'Complete tasks'),(29,41,'Test transfer'),(41,52,'Keep the failures'),(52,60,'The next milestone')]
    spec=dict(title='One policy. Different bodies. Learning how to move.',duration_seconds=60,fps=FPS,resolution=[W,H],audio='Silent; captions carry the narrative.',scenes=scenes,sources=sources,
      disclosures='Concept graphics are illustrative, not model internals. Simulator clips are cropped excerpts from verified executions at1x simulation playback; terminal frames may hold. No altered trajectories or generated robot footage. E111/E112 aggregate results shown; final card adds E113 release extension14/24 to16/24 on reused development bodies. Model-based selector uses privileged exact physics/state; not real-time or history-only. Goal is future capability, not achieved claim. All E112 gains occur on one seven-joint body.')
    (ROOT/'storyboard.json').write_text(json.dumps(spec,indent=2))
    def clip(im,d,key,rect,t,span,label):
        frames=clips[key];duration=len(frames)/6.25;offset=max(0,duration-span);idx=min(len(frames)-1,int((offset+t)*6.25))
        frame=frames[idx].crop((45,65,595,465));x,y,w,h=rect
        frame=ImageOps.fit(frame,(w,h),method=Image.Resampling.LANCZOS);im.paste(frame,(x,y));d.rounded_rectangle((x,y,x+w,y+h),radius=3,outline=(55,84,108),width=2)
        text(d,(x+14,y+12),label,20,CYAN,True)
    writer=imageio.get_writer(ROOT/'progress_demo.mp4',fps=FPS,codec='libx264',quality=8,macro_block_size=16)
    for f in range(60*FPS):
        t=f/FPS;scene=next(i for i,(a,b,_) in enumerate(scenes) if a<=t<b);local=t-scenes[scene][0]
        im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
        for x in range(0,W,64):d.line((x,0,x,H),fill=(15,25,40))
        for y in range(0,H,64):d.line((0,y,W,y),fill=(15,25,40))
        text(d,(48,25),'EASE  /  RESEARCH IN PROGRESS',18,CYAN,True);text(d,(1000,26),f'{scene+1:02d} / 06',18,MUTED)
        d.line((48,66,1232,66),fill=(48,69,90),width=1)
        if scene==0:
            text(d,(48,105),'One policy.',62,WHITE,True);text(d,(48,181),'Different bodies.',62,CYAN,True)
            text(d,(52,285),'The research goal: adapt from motion history',28)
            text(d,(52,329),'to control unfamiliar robot arms.',28,MUTED)
            for k,n in enumerate((6,7)):
                ox=770+k*235;oy=480;x,y=ox,oy;pts=[(x,y)]
                for j in range(n):
                    a=-1.5+.65*math.sin(t*.5+j*.8);x+=48*math.cos(a);y+=48*math.sin(a);pts.append((x,y))
                d.line(pts,fill=CYAN if k==0 else AMBER,width=9)
                for x,y in pts:d.ellipse((x-7,y-7,x+7,y+7),fill=WHITE)
                text(d,(ox-12,525),f'{n} joints',24,MUTED)
            text(d,(52,535),'GOAL / illustrative motion graphics',17,AMBER)
        elif scene==1:
            text(d,(48,103),'Use what the robot just did.',43,WHITE,True)
            text(d,(48,167),'Actions + observed motion provide local response information.',25,MUTED)
            labels=['RECENT HISTORY','LEARNED RESPONSE','NEXT ACTION']
            for i,label in enumerate(labels):
                x=48+408*i;box(d,(x,252,x+370,462));text(d,(x+22,278),label,20,CYAN,True)
                if i==0:
                    for j in range(9):
                        h=25+65*(.5+.5*math.sin(local*1.4+j));d.rectangle((x+25+j*35,420-h,x+45+j*35,420),fill=CYAN if j%2 else (43,104,135))
                elif i==1:
                    for j in range(6):
                        for k in range(7):
                            c=int(65+70*(.5+.5*math.sin(local+j+k)));d.rectangle((x+35+k*40,325+j*17,x+64+k*40,337+j*17),fill=(35,c,145))
                else:
                    for j in range(6):d.line((x+40,330+j*18,x+40+int(140+80*math.sin(local+j)),330+j*18),fill=AMBER,width=7)
                if i<2:text(d,(x+378,345),'>',26,MUTED)
            text(d,(48,512),'CURRENT SYSTEM',19,AMBER,True)
            text(d,(48,551),'Frozen neural response estimator + exact-physics action search.',25)
            text(d,(48,592),'Concept schematic. Exact simulator access is still required.',21,MUTED)
        elif scene in (2,3):
            train=scene==2
            text(d,(48,94),'From safer actions to complete tasks.' if train else 'Then freeze it. Test different bodies.',38,WHITE,True)
            text(d,(48,148),'Training-body suite: 22/36 -> 31/36 successes' if train else 'New body parameters: 28/48 -> 31/48 successes',29,CYAN,True)
            clip(im,d,'train_old' if train else 'transfer_old',(48,210,570,332),local,12,'BASELINE / failure')
            clip(im,d,'train_new' if train else 'transfer_new',(662,210,570,332),local,12,'PHYSICS SELECTOR / success')
            text(d,(48,560),'Verified end-of-episode excerpts  /  1x simulation playback',18,MUTED)
            text(d,(48,599),'Nine recoveries. No lost successes.' if train else 'Three recoveries. All on one seven-joint body. No regressions.',24,WHITE,True)
            text(d,(48,635),'Familiar bodies; reused placements.' if train else 'Eight new bodies; axis families previously studied. Exact physics remains privileged.',18,AMBER)
        elif scene==4:
            text(d,(48,100),'E112: keep the failures in the picture.',42,WHITE,True)
            clip(im,d,'failure',(48,196,640,385),local,11,'SIX-JOINT / remaining failure')
            text(d,(735,216),'6-joint transfer',28,CYAN,True);text(d,(735,267),'14/24 -> 14/24',35,WHITE,True)
            text(d,(735,334),'No improvement yet.',25,AMBER,True)
            text(d,(735,392),'Local safety is not enough',23);text(d,(735,429),'to guarantee task success.',23)
            text(d,(48,612),'Broad searches take ~520 ms for a 40 ms control interval: not real-time ready.',23,MUTED)
        else:
            text(d,(48,105),'The next milestone',48,WHITE,True)
            text(d,(48,174),'Latest: release extension 14/24 -> 16/24, no regressions (E113).',24,CYAN)
            for i,(a,b) in enumerate([('01','Reliable six-joint transfer'),('02','Separate history learning from physics assistance'),('03','Faster decisions. New independent tests.')]):
                y=230+i*106;box(d,(48,y,1232,y+83));text(d,(72,y+21),a,27,CYAN,True);text(d,(145,y+24),b,28)
            text(d,(48,587),'A reproducible prototype. An open research question.',27,AMBER,True)
            text(d,(48,631),'E113 uses reused development bodies  /  reports + success and failure videos saved',18,MUTED)
        d.rectangle((0,H-7,int(W*t/60),H),fill=CYAN)
        writer.append_data(np.asarray(im))
        if f in (3*FPS,12*FPS,23*FPS,35*FPS,46*FPS,56*FPS):im.save(ROOT/f'preview_{scene+1}.png')
        if f%240==0:print('demo seconds',int(t),flush=True)
    writer.close();reader=imageio.get_reader(ROOT/'progress_demo.mp4');assert reader.count_frames()==60*FPS;reader.close()
    (ROOT/'verification.json').write_text(json.dumps(dict(frames=1440,duration_seconds=60,source_videos_verified=True,sha256=fingerprint(ROOT/'progress_demo.mp4')),indent=2))
    (ROOT/'index.html').write_text('<html><body style="background:#0a101d;color:#edf3fc;font:18px sans-serif;padding:32px"><h1>EASE / Research in progress</h1><video controls width="100%" style="max-width:1280px" src="progress_demo.mp4"></video><p>60-second silent, captioned concept and progress reel. Actual verified simulator excerpts, plus labeled illustrative graphics. Exact physics-assisted results through E112, plus the E113 release extension on the final card.</p><a style="color:#4be2dc" href="storyboard.json">Storyboard and footage provenance</a></body></html>')

if __name__=='__main__':run()
