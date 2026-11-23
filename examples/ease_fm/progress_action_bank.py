"""Frozen E110 action bank,40ms only; no filesystem effects."""
import numpy as np
import mujoco
from .geometry_filter import allowed_pair
from .physics_prediction_audit import forbidden

def search_actions(env,original,aperture,bias,entry,key):
    case,seed,step=key;q=env.observe()['q'];b=np.clip(bias,-.04,.04);fingers=env.data.qpos[env.n:env.n+2].copy()
    goal=np.r_[env.goal,[0.,0.,.15]]
    vector=lambda d:np.r_[d.site_xpos[env.site],.15*d.site_xmat[env.site].reshape(3,3)[:,2]]
    start=vector(env.data);before=float(np.sum((goal-start)**2));minimum_gain=max(1e-8,.01*before)
    data=mujoco.MjData(env.model);geometry=mujoco.MjData(env.geometry_model);cache={}
    def simulate(action):
        action=np.clip(action,-.04,.04);cache_key=action.tobytes()
        if cache_key in cache:return cache[cache_key]
        mujoco.mj_copyData(data,env.model,env.data);target=np.clip(q+action,-2.5,2.5);data.ctrl[:env.n]=target;data.ctrl[env.n:]=aperture
        static_clearance=min(env.clearance(q+f*(target-q),fingers+f*(aperture-fingers)) for f in (.5,1.))
        static_ok=static_clearance>=min(.002,entry['current_clearance'])-1e-6
        hits=set();clearance=.002;result=dict(action=action.tolist(),static_clearance=static_clearance,static_ok=static_ok)
        for k in range(1,21):
            mujoco.mj_step(env.model,data);hits.update(forbidden(env.model,data,env.gids))
            # Geometry-only copy includes the predicted object/finger state;
            # never integrates the expanded-margin model.
            mujoco.mj_copyData(geometry,env.geometry_model,data);mujoco.mj_forward(env.geometry_model,geometry)
            distances=[float(c.dist) for c in geometry.contact[:geometry.ncon] if not allowed_pair({int(c.geom1),int(c.geom2)},env.gids)]
            clearance=min([clearance,*distances])
            if k%20==0:mujoco.mj_forward(env.model,data)
            if k==20:
                v=vector(data);error=float(np.sum((goal-v)**2));gain=before-error
                result[str(k*2)]=dict(error_sq=error,gain=gain,progress=gain>=minimum_gain,collision=bool(hits),pairs=[list(p) for p in sorted(hits)],minimum_clearance=clearance,margin_2mm=clearance>=.002-1e-6,vector=v.tolist(),object=data.xpos[env.obj].tolist(),qpos=data.qpos.tolist(),qvel=data.qvel.tolist())
        cache[cache_key]=result;return result
    candidates=[]
    def add(name,group,action):candidates.append(dict(name=name,group=group,**simulate(action)))
    add('original','original',original)
    for name,a in [('half',b+.5*(original-b)),('quarter',b+.25*(original-b)),('bias',b),('hold',np.zeros(env.n)),('brake',b-.04*env.data.qvel[:env.n])]:add(name,'narrow',np.clip(a,-.04,.04))
    for amplitude in (.01,.02,.04):
        for j in range(env.n):
            for sign in (-1,1):
                a=b.copy();a[j]+=sign*amplitude;add(f'axis{j}_{sign}_{amplitude}','axis',np.clip(a,-.04,.04))
    rng=np.random.default_rng(11000000+case*100000+seed*10000+step)
    for i in range(64):
        direction=rng.normal(size=env.n);direction/=np.linalg.norm(direction)
        add(f'random{i}','random',np.clip(b+(.02 if i%2==0 else .04)*direction,-.04,.04))
    # Model-based finite-difference response, not analytic task Jacobians or
    # a learned dynamics prediction. Probes are already in the axis bank.
    v0=np.array(simulate(b)['40']['vector']);jac=np.zeros((6,env.n))
    for j in range(env.n):
        plus=b.copy();minus=b.copy();plus[j]+= .01;minus[j]-=.01;plus=np.clip(plus,-.04,.04);minus=np.clip(minus,-.04,.04)
        jac[:,j]=(np.array(simulate(plus)['40']['vector'])-np.array(simulate(minus)['40']['vector']))/(plus[j]-minus[j])
    correction=jac.T@np.linalg.solve(jac@jac.T+.015**2*np.eye(6),.35*(goal-v0))
    for scale in (.25,.5,1.):add(f'directed{scale}','directed',np.clip(b+scale*correction,-.04,.04))
    return dict(candidates=candidates,unique_physics_branches=len(cache),before_error_sq=before)
