"""Build E130 from the audited E129 fitting-family comparison workflow."""
from pathlib import Path

def run():
    root=Path(__file__).parent
    s=(root/'paired_advantage_objective.py').read_text()
    s=s.replace('E129','E130').replace('paired_advantage_objective','richer_task_state').replace('ease_richer_task_state_v1','ease_richer_task_state_v1')
    s=s.replace('class Advantage','class RichGate').replace('Advantage()','RichGate()')
    s=s.replace('nn.Linear(139,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,1),nn.Tanh()', 'nn.Linear(162,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,2)')
    s=s.replace('return self.net(x).squeeze(-1)','return self.net(x)').replace('m(xx).numpy()','m(xx).sigmoid().numpy()')
    start=s.index('def prepare():');end=s.index('\ndef train():')
    extra=(root/'richer_task_state_features.py').read_text()
    s=s[:start]+extra[extra.index('def extract('):]+s[end:]
    s=s.replace("torch.set_num_threads(4)\n    x,y,folds,meta,tasks=data();target=y[:,1]-y[:,0]", "torch.set_num_threads(4)\n    x,y,folds,meta,tasks=data();target=y")
    s=s.replace('oof=np.empty((3,len(x)),np.float32)','oof=np.empty((3,len(x),2),np.float32)')
    s=s.replace("n=np.load(PRIOR/'models'/f'fold_{fold}'/'normalization.npz');mean,scale=n['mean'],n['scale']", "n=np.load(PRIOR/'models'/f'fold_{fold}'/'normalization.npz');mean=np.r_[n['mean'],x[train_ids,139:].mean(0)];scale=np.r_[n['scale'],np.maximum(x[train_ids,139:].std(0),1e-6)]")
    # Preserve established feature normalization exactly; added features use their own fitting rows.
    s=s.replace("np.testing.assert_array_equal(mean,x[train_ids].mean(0));np.testing.assert_array_equal(scale,np.maximum(x[train_ids].std(0),1e-6))", "np.testing.assert_array_equal(mean[:139],n['mean']);np.testing.assert_array_equal(scale[:139],n['scale'])")
    a=s.index('            for layer in (0,2):');b=s.index('            opt=',a)
    s=s[:a]+'''            with torch.no_grad():
                m.net[0].weight[:,:139].copy_(original.net[0].weight)
                m.net[0].weight[:,139:].zero_()
                m.net[0].bias.copy_(original.net[0].bias)
                for layer in (2,4):m.net[layer].load_state_dict(original.net[layer].state_dict())
            for layer in (2,4):
                torch.testing.assert_close(m.net[layer].weight,original.net[layer].weight,rtol=0,atol=0)
            torch.testing.assert_close(m.net[0].weight[:,:139],original.net[0].weight,rtol=0,atol=0)
            assert torch.count_nonzero(m.net[0].weight[:,139:])==0
''' + s[b:]
    s=s.replace('nn.functional.mse_loss(m(xx),yy)','nn.functional.binary_cross_entropy_with_logits(m(xx),yy)')
    s=s.replace('advantages=oof,row_fold=folds','advantages=oof[:,:,1]-oof[:,:,0],probabilities=oof,row_fold=folds')
    s=s.replace('training_mse=losses','training_bce=losses').replace('hidden_initialization_exact=True','original_weights_initialization_exact=True,added_weights_initially_zero=True')
    # The shared workflow names the second condition direct; rename to the actual ablation.
    s=s.replace('direct','richer')
    s=s.replace('paired_advantage_videos','richer_task_state_videos').replace('Paired-advantage objective comparison','Richer measured task-state comparison').replace('richerory','directory')
    s=s.replace("np.testing.assert_array_equal(n['mean'],x[ids].mean(0));np.testing.assert_array_equal(n['scale'],np.maximum(x[ids].std(0),1e-6))", "np.testing.assert_array_equal(n['mean'][139:],x[ids,139:].mean(0));np.testing.assert_array_equal(n['scale'][139:],np.maximum(x[ids,139:].std(0),1e-6))")
    s=s.replace("np.testing.assert_array_equal(predict(models,n['mean'],n['scale'],x[val]),richer[:,val]);np.testing.assert_array_equal(probabilities(original,n['mean'],n['scale'],x[val]),two[:,val])", "rp=predict(models,n['mean'],n['scale'],x[val]);np.testing.assert_array_equal(rp,np.load(ROOT/'richer_oof.npz')['probabilities'][:,val]);np.testing.assert_array_equal(rp[:,:,1]-rp[:,:,0],richer[:,val]);np.testing.assert_array_equal(probabilities(original,n['mean'][:139],n['scale'][:139],x[val,:139]),two[:,val])")
    s=s.replace("save(ROOT/'verification.json',dict(", "verify_features()\n    save(ROOT/'verification.json',dict(measured_features_reconstructed=True,")
    s=s.replace("for name in ('protocol.json','richer_oof.npz','training.json','folds.json')", "for name in ('protocol.json','richer_oof.npz','training.json','folds.json','features.npz','feature_provenance.json')")
    s=s.replace('Direct paired advantage versus two success heads','Richer measured task state with two success heads')
    (root/'richer_task_state.py').write_text(s)

if __name__=='__main__':run()
