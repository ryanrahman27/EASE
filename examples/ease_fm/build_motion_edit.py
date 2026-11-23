"""Create an isolated motion-design edit from the verified physical replay."""
from pathlib import Path

p=Path(__file__).with_name('clean_task_video.py')
s=p.read_text()
s=s.replace("ROOT=Path('outputs/ease_known_arm_task_demo_v2')", "SOURCE=Path('outputs/ease_known_arm_task_demo_v2')\nROOT=Path('outputs/ease_known_arm_task_demo_v3');ROOT.mkdir(exist_ok=True)")
s=s.replace("reader=imageio.get_reader(ROOT/f'{name}_physical.mp4')", "reader=imageio.get_reader(SOURCE/f'{name}_physical.mp4')")
s=s.replace("traces[name]=json.loads((ROOT/f'{name}_attempt1.json')", "traces[name]=json.loads((SOURCE/f'{name}_attempt1.json')")
s=s.replace("json.loads((ROOT/f'{n}_clearance.json')", "json.loads((SOURCE/f'{n}_clearance.json')")
start=s.index('        if at and t<captions[at][0]+.6:')
stop=s.index('        im.paste(layer.crop',start)
s=s[:start]+'''        if at and t<captions[at][0]+.6:
            u=smooth((t-captions[at][0])/.6);layer=Image.new('RGB',(1280,720),'black')
            ld=ImageDraw.Draw(layer)
            center(ld,630-int(48*u),captions[at-1][1],25)
            center(ld,630+int(48*(1-u)),captions[at][1],25)
'''+s[stop:]
start=s.index('        if t>=20:im=')
stop=s.index('        writer.append_data',start)
s=s[:start]+'''        if t>=20:im=scan_panels(im,card,(t-20)/1.15)
        if t>=26:im=joint_iris(im,end,(t-26)/1.1)
        if t<.8:im=aperture(im,t/.8)
'''+s[stop:]
s=s.replace('(175,350,600,800,1150,1400)', '(15,175,215,350,600,800,1015,1025,1040,1150,1310,1325,1340,1400)')
s=s.replace('Soft editorial dissolves.', 'Staggered scanning panels, a two-link iris, split-aperture opening, and masked rolling captions. No fades; complete task footage unchanged.')
s=s.replace("for file in ('build_clean_task_demo.py','known_arm_task_clean.py','clean_task_video.py','task_mesh_clearance.py')", "for file in ('build_motion_edit.py','motion_task_video.py')")
helpers='''
def aperture(im,u):
    if u>=1:return im
    width=int(640*smooth(u));out=Image.new('RGB',im.size,'black')
    out.paste(im.crop((640-width,0,640+width,720)),(640-width,0))
    d=ImageDraw.Draw(out)
    for x in (640-width,640+width):
        d.line((x,0,x,720),fill=GREEN,width=2)
        d.ellipse((x-5,355,x+5,365),fill=GREEN)
    return out

def scan_panels(old,new,u):
    # Three staggered shutters echo the three benchmark columns.
    if u>=1:return new.copy()
    out=old.copy();d=ImageDraw.Draw(out)
    for k,(left,right) in enumerate(((0,432),(432,848),(848,1280))):
        v=smooth((u-k*.11)/.78);y=int(720*(1-v))
        if y<720:out.paste(new.crop((left,y,right,720)),(left,y))
        if 0<v<1:
            d.line((left,y,right,y),fill=GREEN,width=3)
            x=(left+right)//2
            d.ellipse((x-6,y-6,x+6,y+6),fill=GREEN)
            # Quiet trailing ticks give the sweep a measurement-instrument feel.
            for tick in range(left+24,right,40):d.line((tick,y,tick,y+8),fill=GREEN,width=1)
    return out

def joint_iris(old,new,u):
    # A two-link graphic unfolds around the centre, opening a circular aperture.
    # This is editorial geometry, never overlaid during robot execution.
    if u>=1:return new.copy()
    import math
    v=smooth(u);r=int(760*v)
    mask=Image.new('L',old.size,0);md=ImageDraw.Draw(mask)
    md.ellipse((640-r,360-r,640+r,360+r),fill=255)
    out=Image.composite(new,old,mask);d=ImageDraw.Draw(out)
    if r>0:d.ellipse((640-r,360-r,640+r,360+r),outline=GREEN,width=3)
    # Linkage stays outside the reveal, then exits with its expanding boundary.
    a=-math.pi/2+v*math.pi*1.3
    pts=[]
    for offset,scale in ((0,1),(.48,1.10),(.94,1.02)):
        pts.append((int(640+(r+22)*scale*math.cos(a+offset)),int(360+(r+22)*scale*math.sin(a+offset))))
    d.line(pts,fill=GREEN,width=3)
    for x,y in pts:d.ellipse((x-6,y-6,x+6,y+6),fill=(0,0,0),outline=GREEN,width=3)
    return out

'''
s=s.replace('def compose():',helpers+'def compose():')
Path(__file__).with_name('motion_task_video.py').write_text(s)
