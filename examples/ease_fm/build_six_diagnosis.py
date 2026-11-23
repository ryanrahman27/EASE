"""Instrument E115 without changing baseline actions."""
from pathlib import Path
b=Path(__file__).parent;s=(b/'frozen_bias_core.py').read_text()
s=s.replace('ease_frozen_bias_v1','ease_six_motion_diagnosis_v1')
s=s.replace('save_states=False):','save_states=False,condition="baseline"):')
s=s.replace('telemetry=[];states=[];updates=0;active=False','telemetry=[];states=[];updates=0;active=False;diagnostics=[];swapped=False')
s=s.replace('    class Controller(LocalPose):\n', '''    class Controller(LocalPose):
        def predict(self):
            nonlocal swapped
            if active and condition!='baseline' and env.phase==condition:swapped=True
            saved=self.model
            if swapped:self.model=None
            try:return super().predict()
            finally:self.model=saved
''')
needle="            telemetry.append(dict(phase=env.phase,position_prediction_error="
idx=s.index(needle)
s=s[:idx]+'''            saved=self.model
            self.model=model
            try:jl,_=LocalPose.predict(self)
            finally:self.model=saved
            ja=np.linalg.solve(self.gram,self.cross).T
            error=np.r_[env.goal-before['position'],ROT_SCALE*(np.array([0.,0.,1.])-before['rotation'][:,2])]
            def action(j):return np.clip(j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),.35*error)+self.bias,-.04,.04)
            al=action(jl);aa=action(ja);eig=np.linalg.eigvalsh(self.gram)
            diagnostics.append(dict(phase=env.phase,learned_xyz_error=float(np.linalg.norm(dx[:3]-(jl@dq)[:3])),analytical_xyz_error=float(np.linalg.norm(dx[:3]-(ja@dq)[:3])),learned_axis_error=float(np.linalg.norm(dx[3:]-(jl@dq)[3:])),analytical_axis_error=float(np.linalg.norm(dx[3:]-(ja@dq)[3:])),relative_matrix_difference=float(np.linalg.norm(jl-ja)/(np.linalg.norm(ja)+1e-12)),action_difference=float(np.linalg.norm(al-aa)),learned_saturated=int(np.sum(np.abs(al)>=.04-1e-10)),analytical_saturated=int(np.sum(np.abs(aa)>=.04-1e-10)),gram_condition=float(eig[-1]/eig[0]),gram_min_eigenvalue=float(eig[0]),error_norm=float(np.linalg.norm(error)),swapped=swapped))
'''+s[idx:]
s=s.replace('env.phase=name\n','env.phase=name;env.goal=goal\n')
s=s.replace("yield 'probe',env.observe()['position'].copy(),0.,32", "env.goal=env.observe()['position'].copy()\n                    yield 'probe',env.goal,0.,32")
s=s.replace("stem=f'{case}_{method}_{mode}'", "stem=f'{case}_{method}_{condition}'")
s=s.replace('    return result\n',"    result.update(condition=condition,diagnostics=diagnostics)\n    return result\n")
(b/'six_motion_core.py').write_text(s)
