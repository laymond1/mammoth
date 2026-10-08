"""Synthetic CPU checks for the fixed-size, inference-only PROL diagnosis."""
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from scripts.cl_runtime import configure_native_environment
configure_native_environment()

import numpy as np
import pytest
import torch
from scripts import diagnose_prol_checkpoint as diagnostic
from scripts import prol_diagnostic_protocol as shared
from scripts import prol_reference_utils as reference
from scripts import run_prol_worker as training_worker


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv('MAMMOTH_DEVICE','cpu')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    previous=torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def code():
    if not shared.REFERENCE_SOURCE.is_dir():
        pytest.skip('Optional pinned PROL source is absent')
    return reference.load_author_code(shared.REFERENCE_SOURCE)


def tiny_author(code):
    return reference.build_author_branches(code,num_classes=100,embed_dim=12,depth=2,num_heads=3,
        img_size=8,patch_size=4,prompt_layers=[0,1],n_tasks=10)


def toy_frozen_hashes(branches):
    return {branch:{'sha256':reference.state_digest({name:parameter for name,parameter in module.named_parameters()
        if not reference.learned_name(branch,name)}),'tensors':len([name for name,_ in module.named_parameters()
        if not reference.learned_name(branch,name)])} for branch,module in branches.items()}


def payload(branches):
    state=reference.learned_state(branches)
    return {'task':1,'seen_classes':10,'state':state,'learned_sha256':reference.state_digest(state),
        'prompt_lifecycle':{'old_num_k':0,'new_num_k':1,'old_num_c':0,'new_num_c':10,'kmax_list':[1],'lmax_list':[10]}}


def record():
    return {'counts':{'train_steps':0,'backward_calls':0,'optimizer_updates':0,'query_forwards':0,
        'prompt_forwards':0,'model_evaluations':0,'encoder_forward_calls':0,'diagnostic_image_visits':0},'matrix':[]}


def recorder(tmp_path):
    return SimpleNamespace(output=tmp_path,phase=lambda _:nullcontext(),active_phase='synthetic',snapshots={})


def test_import_and_worker_guard_start_no_gpu_logging_or_training(monkeypatch,tmp_path):
    for name in ['CL_PROL_DIAGNOSTIC_EXECUTE','CUDA_VISIBLE_DEVICES','WANDB_MODE']:
        monkeypatch.delenv(name,raising=False)
    with pytest.raises(ValueError,match='explicit GPU1'):
        diagnostic.worker(tmp_path/'absent.json')


def test_sigterm_uses_caught_failure_path_and_restores_previous_handler(monkeypatch,tmp_path):
    import signal
    previous=signal.getsignal(signal.SIGTERM)
    observed=[]
    def terminate(path):
        observed.append(path)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
    monkeypatch.setattr(diagnostic,'worker',terminate)
    with pytest.raises(KeyboardInterrupt,match='termination requested'):
        diagnostic.main(['--worker',str(tmp_path/'spec.json')])
    assert observed==[tmp_path/'spec.json']
    assert signal.getsignal(signal.SIGTERM) is previous


def test_historical_callback_has_unchanged_ast_and_bound_reference():
    callback,digest=diagnostic.historical_callback()
    assert callback.__globals__['reference'] is reference
    assert len(digest)==64 and callback.__name__=='prompt_output'
    assert shared.ORIGIN_EXECUTION_COMMIT in callback.__code__.co_filename


def test_restore_catalogue_copies_persisted_norms_heads_and_lifecycle(code):
    branches=tiny_author(code)
    saved=payload(branches)
    saved['state']['prompt.norm.weight'].add_(.125)
    saved['learned_sha256']=reference.state_digest(saved['state'])
    before=training_worker.rng_digest(torch)
    diagnostic.restore_reference(branches,saved)
    assert training_worker.rng_digest(torch)==before
    assert reference.state_digest(reference.learned_state(branches))==saved['learned_sha256']
    assert branches['prompt'].e_prompt.lmax_list==[10]
    assert branches['query'].training and not branches['prompt'].training
    assert all(parameter.grad is None for module in branches.values() for parameter in module.parameters())
    assert all(parameter.requires_grad for parameter in branches['prompt'].e_prompt.k_comp_gen.parameters())
    bad=deepcopy(saved)
    bad['prompt_lifecycle']['lmax_list']=[20]
    with pytest.raises(ValueError,match='lifecycle'):
        diagnostic.restore_reference(branches,bad)
    bad=deepcopy(saved)
    del bad['state']['prompt.norm.weight']
    bad['learned_sha256']=reference.state_digest(bad['state'])
    with pytest.raises(ValueError,match='catalogue'):
        diagnostic.restore_reference(branches,bad)


def test_exact40_evaluations_match_with_zero_backward_and_no_off_hooks(monkeypatch,tmp_path,code):
    torch.manual_seed(1993)
    branches=tiny_author(code)
    saved=payload(branches)
    diagnostic.restore_reference(branches,saved)
    clones,_=training_worker.compact_clone(branches,saved)
    monkeypatch.setattr(training_worker,'frozen_encoder_hashes',toy_frozen_hashes)
    callback,_=diagnostic.historical_callback()
    generator=torch.Generator().manual_seed(55)
    inputs=torch.randn(40,3,8,8,generator=generator)
    labels=np.arange(40,dtype=np.int64)%10
    ids=np.arange(40,dtype=np.int64)
    captures=[]
    original=diagnostic.evaluate_pair
    def evaluate(branches,inputs,counts):
        captures.append(len(branches['prompt'].e_prompt._forward_hooks))
        assert not torch.is_grad_enabled()
        return original(branches,inputs,counts)
    monkeypatch.setattr(diagnostic,'evaluate_pair',evaluate)
    value=record()
    matrix=diagnostic.collect_matrix({'author_restored':branches,'compact_clone':clones},inputs,labels,ids,
        np.zeros(40,dtype=np.int64),tmp_path,recorder(tmp_path),value,callback,
        audit_protocol={'prompt_layers':[0,1],'tasks':2,'n_epochs':1})
    assert len(matrix)==40
    assert captures==[0]*10+[1]*10+[0]*10+[1]*10
    assert value['counts']=={'train_steps':0,'backward_calls':0,'optimizer_updates':0,'query_forwards':40,
        'prompt_forwards':40,'model_evaluations':40,'encoder_forward_calls':80,'diagnostic_image_visits':1600}
    assert all(row['rng_unchanged'] and row['learned_unchanged'] and row['frozen_unchanged'] for row in matrix)
    assert all(row['finite'] and row['gradients_absent'] for row in matrix)
    assert all(not module.e_prompt._forward_hooks for module in [branches['prompt'],clones['prompt']])
    assert len(list(tmp_path.glob('trace_*.npz')))==20
    with np.load(tmp_path/'forward_outputs.npz',allow_pickle=False) as outputs:
        assert outputs['logits'].shape==(40,40,10)
        assert outputs['query_features'].shape==(40,40,12)
        assert outputs['rng_cpu_before'].shape[0]==40
        assert np.array_equal(outputs['rng_cpu_before'],outputs['rng_cpu_after'])
        assert all(np.array_equal(outputs['logits'][0],item) for item in outputs['logits'])
    with np.load(tmp_path/'trace_10.npz',allow_pickle=False) as trace:
        assert trace['layer0_similarity_new'].shape==(40,10)
        assert trace['layer0_indices'].shape==(40,1)
        assert trace['layer0_prefix'].shape==(40,2,5,3,4)
        assert 'layer1_post_affine_scaler' in trace


def test_on_trace_preserves_preclamp_storage_and_records_actual_clamp(monkeypatch,tmp_path,code):
    branches=tiny_author(code)
    diagnostic.restore_reference(branches,payload(branches))
    monkeypatch.setattr(training_worker,'frozen_encoder_hashes',toy_frozen_hashes)
    callback,_=diagnostic.historical_callback()
    module=branches['prompt'].e_prompt
    with torch.no_grad():
        module.scaler.fill_(2)
        module.shifter.fill_(1)
    with diagnostic.audited_forward(branches,recorder(tmp_path),record(),torch,np,callback,{'prompt_layers':[0,1]}) as (audit,trace),torch.no_grad():
        diagnostic.evaluate_pair(branches,torch.randn(40,3,8,8),record()['counts'])
    assert len(trace)==2
    assert torch.all(trace[0]['pre_affine']['scaler']==2)
    assert torch.all(trace[0]['pre_affine']['shifter']==1)
    assert torch.all(trace[0]['post_affine']['scaler']==1.001)
    assert torch.all(trace[0]['post_affine']['shifter']==.0001)
    assert trace[0]['pre_affine']['scaler'].data_ptr()!=trace[0]['post_affine']['scaler'].data_ptr()
    assert audit.routing_layer_count==2 and len(audit.eval_routes)==1


def test_forbid_training_guards_backward_and_optimizer():
    value=record()['counts']
    tensor=torch.tensor(1.,requires_grad=True)
    optimizer=torch.optim.Adam([tensor],lr=.01)
    with diagnostic.forbid_training(torch,value):
        with pytest.raises(RuntimeError,match='backward'):
            tensor.backward()
        with pytest.raises(RuntimeError,match='optimizer'):
            optimizer.step()
    assert value['backward_calls']==value['optimizer_updates']==1


def test_checkpoint_failure_records_separate_rng_and_output_proofs(monkeypatch,tmp_path):
    """Use the actual worker's checkpoint path with a deliberate probe difference."""
    from tests.test_prol_worker import small_native,base_record,toy_frozen_hash
    branches,network=small_native()
    monkeypatch.setattr(training_worker,'frozen_encoder_hashes',toy_frozen_hash)
    network.begin_task(0,10)
    network.task_updates[0]=1
    network.finalize_task()
    inputs=torch.randn(40,3,8,8)
    with torch.no_grad():
        logits=network.outputs(inputs)['logits'].detach().clone()
    value=base_record()
    local_recorder=recorder(tmp_path)
    audit=training_worker.ProlAudit(branches,SimpleNamespace(),local_recorder,value,torch,np)
    audit.task=0
    audit.task_old={}
    # Initial frozen catalogue captures the intended unchanged encoder values.
    audit.last_probe=(inputs,logits+1)
    with pytest.raises(RuntimeError,match='clamped outputs or RNG'):
        audit.finalized()
    proof=value['checkpoint_probe_audits'][0]
    assert proof['checks']['disk_catalogue_matches']
    assert proof['checks']['clone_catalogue_matches']
    assert proof['checks']['clone_rng_unchanged']
    assert proof['checks']['probe_outputs_bitexact'] is False
    assert proof['checks']['probe_rng_unchanged'] is True
    assert proof['max_absolute_logit_difference']>0
    assert proof['predictions_changed']==0
    assert (tmp_path/'task1_checkpoint_probe_failure.npz').is_file()
    assert (tmp_path/'checkpoint_probe_audits.json').is_file()
    audit.close()


def test_validation_error_after_full_matrix_publishes_failed_status_only(monkeypatch,tmp_path,code):
    """Mock external services/device placement; no GPU/network work is permitted."""
    import sys
    branches=tiny_author(code)
    saved=payload(branches)
    saved['frozen_encoder']=toy_frozen_hashes(branches)
    checkpoint=tmp_path/'checkpoint.pt'
    torch.save(saved,checkpoint)
    spec={'name':'synthetic-post-matrix-failure','output':str(tmp_path),'data_root':'synthetic',
        'checkpoint_file':'synthetic.npz','prior_result':'synthetic-result.json','prior_result_sha256':'a'*64,
        'prior_predictions':'synthetic-predictions.npz','prior_predictions_sha256':'a'*64,
        'prior_checkpoint':str(checkpoint),'prior_checkpoint_sha256':'a'*64,
        'reference_source':'synthetic','reference_source_sha256':{},'execution_source_sha256':{}}
    spec_path=tmp_path/'spec.json'
    spec_path.write_text(json.dumps(spec))
    for key,value in [('CL_PROL_DIAGNOSTIC_EXECUTE','1'),('CUDA_VISIBLE_DEVICES','1'),('WANDB_MODE','online')]:
        monkeypatch.setenv(key,value)
    monkeypatch.setattr(torch.cuda,'_lazy_init',Mock(side_effect=AssertionError('Synthetic check must not initialize CUDA')))
    original_device=torch.device
    monkeypatch.setattr(torch,'device',lambda value:original_device('cpu') if value=='cuda:0' else original_device(value))
    fake_wandb=SimpleNamespace(run=None)
    fake_wandb.init=lambda **kwargs:setattr(fake_wandb,'run',SimpleNamespace(url='synthetic-only'))
    monkeypatch.setitem(sys.modules,'wandb',fake_wandb)
    class FakeRecorder:
        active_phase='synthetic'
        def __init__(self,*args,**kwargs):pass
        def start(self):pass
        def phase(self,name):return nullcontext()
        def save(self,value):return {**value,'metrics':{},'phases':{},'measurement_notes':{}}
    monkeypatch.setattr(diagnostic,'Recorder',FakeRecorder)
    monkeypatch.setattr(shared,'validate_spec',lambda _:True)
    monkeypatch.setattr(shared,'verify_prior_inputs',lambda :{})
    monkeypatch.setattr(shared,'execution_sources',lambda :{})
    monkeypatch.setattr(shared,'file_hash',lambda _:'a'*64)
    monkeypatch.setattr(shared,'CHECKPOINT_SHA256','a'*64)
    monkeypatch.setattr(reference,'verify_author_source',lambda *args:True)
    monkeypatch.setattr(reference,'load_author_code',lambda *args:code)
    monkeypatch.setattr(reference,'build_author_branches',lambda *args:branches)
    monkeypatch.setattr(training_worker,'configure_precision',lambda _:shared.PRECISION)
    monkeypatch.setattr(training_worker,'canonical_hashes',lambda _:{'query':shared.CANONICAL_PRETRAINED_SHA256,'prompt':shared.CANONICAL_PRETRAINED_SHA256})
    monkeypatch.setattr(training_worker,'frozen_encoder_hashes',toy_frozen_hashes)
    inputs=torch.zeros(40,3,8,8)
    evidence={'input_sha256':diagnostic.tensor_hash('inputs',inputs)}
    monkeypatch.setattr(diagnostic,'reconstruct_tail',lambda *args:(inputs,np.zeros(40,dtype=np.int64),
        np.arange(40,dtype=np.int64),np.zeros(40,dtype=np.int64),evidence))
    monkeypatch.setattr(diagnostic,'collect_matrix',lambda *args,**kwargs:args[7].update(matrix=[{}]*40))
    monkeypatch.setattr(diagnostic,'validate_saved_record',Mock(side_effect=ValueError('deliberate post-matrix validation failure')))
    published=[]
    monkeypatch.setattr(diagnostic,'upload_record',lambda wandb,output,record:published.append(deepcopy(record)))
    assert diagnostic.worker(spec_path)==1
    assert len(published)==1
    assert published[0]['status']==published[0]['computation_status']=='failed'
    assert published[0]['error_type']=='ValueError'
    assert len(published[0]['matrix'])==40


def test_reconstruct_tail_preserves_original_batch_order_with_pinned_transform(monkeypatch,tmp_path):
    import hashlib
    import pickle
    from torchvision import transforms
    from scripts import hide_protocol
    raw_root=tmp_path/'CIFAR100/cifar-100-python'
    raw_root.mkdir(parents=True)
    ids=np.arange(1000,dtype=np.int64)
    targets=ids//100
    data=np.broadcast_to((ids%256).astype(np.uint8)[:,None],(1000,3072)).copy()
    with (raw_root/'test').open('wb') as stream:
        pickle.dump({'data':data,'fine_labels':targets.tolist()},stream)
    evaluation_ids=np.random.default_rng(1993).permutation(ids)
    predictions=tmp_path/'saved_predictions.npz'
    np.savez_compressed(predictions,global_ids=ids,labels=targets,predictions=targets,
        evaluation_global_ids=evaluation_ids,batch_sizes=np.array([64]*15+[40]))
    transform=transforms.Compose([transforms.Resize(256,interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0,0,0],[1,1,1])])
    data_files={'files':{'test':{'sha256':'a'*64}}}
    monkeypatch.setattr(hide_protocol,'validate_shared_data',lambda _:data_files)
    old={'data_snapshots':{'task1_test':{'transform_repr':repr(transform),
        'transform_repr_sha256':hashlib.sha256(repr(transform).encode()).hexdigest()}},
        'routing_audits':[{'batches':[{'samples':40}]}]}
    inputs,labels,actual_ids,old_predictions,evidence=diagnostic.reconstruct_tail(
        {'data_root':str(tmp_path),'data_files':data_files,'prior_predictions':str(predictions)},old)
    assert inputs.shape==(40,3,224,224)
    assert np.array_equal(actual_ids,evaluation_ids[-40:])
    assert not np.array_equal(actual_ids,np.sort(actual_ids))
    assert np.array_equal(labels,targets[actual_ids])
    assert np.array_equal(old_predictions,targets[actual_ids])
    torch.testing.assert_close(inputs[:,0,0,0],torch.from_numpy((actual_ids%256).astype(np.float32))/255,rtol=0,atol=0)
    assert evidence['input_sha256']==diagnostic.tensor_hash('inputs',inputs)
    bad=deepcopy(old)
    bad['data_snapshots']['task1_test']['transform_repr_sha256']='b'*64
    with pytest.raises(ValueError,match='preprocessing'):
        diagnostic.reconstruct_tail({'data_root':str(tmp_path),'data_files':data_files,'prior_predictions':str(predictions)},bad)


def test_diagnostic_rng_change_is_saved_evidence_and_never_a_gate_relaxation(monkeypatch):
    monkeypatch.setattr(shared,'validate_spec',lambda _:True)
    value=record()
    value.update(status='completed',computation_status='completed',precision=shared.PRECISION,input_sha256='a'*64)
    value['counts'].update(query_forwards=40,prompt_forwards=40,model_evaluations=40,
        encoder_forward_calls=80,diagnostic_image_visits=1600)
    for side in shared.MODEL_SIDES:
        for mode in shared.HOOK_MODES:
            for repeat in range(1,11):
                value['matrix'].append({'row_index':len(value['matrix']),'branch':side,'audit_mode':mode,'repeat':repeat,
                    'input_sha256':'a'*64,'finite':True,'gradients_absent':True,'labels_provided':False,
                    'oracle_task_id':None,'query_training_mode':True,'prompt_training_mode':False,
                    'rng_before_sha256':'b'*64,'rng_after_sha256':'c'*64,'rng_unchanged':False,
                    'learned_before_sha256':'d'*64,'learned_after_sha256':'d'*64,'learned_unchanged':True,
                    'frozen_before':{},'frozen_after':{},'frozen_unchanged':True})
    assert diagnostic.validate_saved_record(value)
    value['matrix'][0]['rng_unchanged']=True
    with pytest.raises(ValueError,match='equality flags'):
        diagnostic.validate_saved_record(value)
