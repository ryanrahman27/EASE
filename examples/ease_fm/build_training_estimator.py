from pathlib import Path
b=Path(__file__).parent;s=(b/'frozen_bias_core.py').read_text().replace('ease_frozen_bias_v1','ease_estimator_repair_v1')
s=s.replace('save_states=False):','save_states=False,contexts=None):')
s=s.replace("            telemetry.append(dict(phase=env.phase,position_prediction_error=",'''            if contexts is not None:
                g=np.zeros((7,7),np.float32);cross=np.zeros((7,6),np.float32);mask=np.arange(7)<self.n;b=np.zeros(7,np.float32);delta=np.zeros(7,np.float32)
                g[:self.n,:self.n]=self.gram;cross[:self.n]=self.cross;b[:self.n]=self.bias;delta[:self.n]=dq
                error=np.r_[env.goal-before['position'],ROT_SCALE*(np.array([0.,0.,1.])-before['rotation'][:,2])]
                contexts.append(dict(g=g,c=cross,mask=mask,error=error.astype(np.float32),bias=b,dq=delta,dx=dx.astype(np.float32),case=case,phase=env.phase,clean=not env.bad,validation=case%3==2))
            telemetry.append(dict(phase=env.phase,position_prediction_error=''')
s=s.replace('env.phase=name\n','env.phase=name;env.goal=goal\n').replace("yield 'probe',env.observe()['position'].copy(),0.,32","env.goal=env.observe()['position'].copy()\n                    yield 'probe',env.goal,0.,32")
(b/'training_estimator_core.py').write_text(s)
