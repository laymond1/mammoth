"""CPU-only checks for actual PROL runners and their resource/audit contracts."""
from argparse import ArgumentParser
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.cl_runtime import configure_native_environment
configure_native_environment()

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from scripts import prol_reference_utils as reference
from scripts import prol_protocol as shared
from scripts import run_prol_worker as worker
from models.prol_prompt_utils.vit import ProlVisionTransformer
from models.prol_prompt_utils.network import ProlNetwork
from models.prol_prompt import ProlPrompt


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv('MAMMOTH_DEVICE', 'cpu')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def code():
    if not shared.REFERENCE_SOURCE.is_dir():
        pytest.skip('Optional pinned source checkout is absent')
    return reference.load_author_code(shared.REFERENCE_SOURCE)


def small_native():
    common = dict(num_classes=100, img_size=8, patch_size=4, embed_dim=12, depth=2, num_heads=3)
    query = ProlVisionTransformer(**common, n_tasks=5, prompted=False)
    prompt = ProlVisionTransformer(**common, n_tasks=10, prompt_length=5, prompt_layers=(0, 1))
    net = ProlNetwork(query, prompt, 10)
    net.pretrained_checkpoint = '/tmp/synthetic-prol-only.npz'
    return {'query': query, 'prompt': prompt}, net


def small_author(code):
    return reference.build_author_branches(code, num_classes=100, img_size=8, patch_size=4,
        embed_dim=12, depth=2, num_heads=3, n_tasks=10, prompt_layers=(0, 1))


def record_for_test(mode='validation', side='native', kind='validation'):
    p = shared.protocol(mode)
    updates = p['tasks'] * 79
    result = {'mode': mode, 'side': side, 'pass': kind, 'method': 'prol-prompt', 'model': 'prol-prompt',
        'protocol': p, 'gpu': '1', 'wandb_entity': 'laymond1', 'wandb_project': 'cl-efficiency',
        'checkpoint_sha256': shared.CHECKPOINT_SHA256, 'canonical_pretrained_sha256': shared.CANONICAL_PRETRAINED_SHA256,
        'reference_commit': shared.REFERENCE_COMMIT, 'training_authorized': True,
        'source_mode': 'official-active-path', 'paper_equivalent': False,
        'measurement_aggregate_eligible': kind == 'measurement', 'status': 'completed', 'computation_status': 'completed',
        'precision': dict(shared.PRECISION), 'structural_checks': {key: True for key in shared.REQUIRED_CHECKS},
        'counts': {'observe_calls': updates, 'optimizer_updates': updates, 'task_begin_extra_samples': 0, 'task_end_extra_samples': 0},
        'epoch_results': [], 'task_results': [], 'first_batch_audits': [], 'prediction_files': [],
        'phases': {}, 'metrics': {'outer_observe_steps': updates, 'outer_training_stream_examples': p['tasks'] * 5000,
                    'evaluation_examples': 1000 if p['tasks'] == 1 else 3000, 'tasks_completed': p['tasks']}}
    for task in range(1, p['tasks'] + 1):
        result['epoch_results'].append({'task': task, 'epoch': 1, 'steps': 79, 'samples': 5000,
            'seed': shared.phase_seed(1993, task - 1, 'train'), 'loss_mean': .3})
        result['first_batch_audits'].append({'task': task, 'epoch': 1})
        result['task_results'].append(worker.exact_counts(np.zeros(task * 1000, dtype=int), np.repeat(np.arange(task * 10), 100), task))
        result['prediction_files'].append({'task': task})
    result['accuracy'] = {'mean_accuracy_pct': result['task_results'][-1]['seen_accuracy_pct']}
    return result


@pytest.mark.parametrize('mode,side,kind', [('validation','author','validation'), ('validation','native','validation'),
    ('efficiency','native','measurement'), ('efficiency','native','flops')])
def test_complete_record_requires_full_workload(mode, side, kind):
    assert worker.validate_saved_record(record_for_test(mode, side, kind))


@pytest.mark.parametrize('mutation', [lambda r: r['counts'].update(optimizer_updates=78),
    lambda r: r['counts'].update(task_begin_extra_samples=1), lambda r: r['metrics'].update(evaluation_examples=1040),
    lambda r: r['epoch_results'][0].update(seed=1993), lambda r: r['epoch_results'][0].update(samples=4999),
    lambda r: r['structural_checks'].update(checkpoint_round_trip=False), lambda r: r['task_results'][0].update(seen_correct=999),
    lambda r: r.update(paper_equivalent=True), lambda r: r['first_batch_audits'].clear()])
def test_partial_changed_workloads_fail(mutation):
    value = record_for_test()
    mutation(value)
    with pytest.raises(ValueError):
        worker.validate_saved_record(value)


def test_initial_catalog_contains_query_head_all_norms_all_prompts(tmp_path):
    branches, _ = small_native()
    original = reference.learned_state(branches)
    assert 'query.head.heads.0.1.weight' in original
    assert 'prompt.blocks.1.norm1.weight' in original
    assert 'prompt.norm.weight' in original
    assert 'prompt.e_prompt.k_comp_gen.0.0.0.weight' in original
    assert 'prompt.e_prompt.scaler' in original
    assert 'prompt.patch_embed.proj.weight' not in original
    path = tmp_path / 'initial.pt'
    metadata = reference.save_initial_state(branches, path, 'a'*64)
    for branch, module in branches.items():
        for name, parameter in module.named_parameters():
            if reference.learned_name(branch, name):
                with torch.no_grad():
                    parameter.add_(2)
    assert reference.load_initial_state(branches, path, 'a'*64) == metadata['learned_sha256']
    assert all(torch.equal(value, reference.learned_state(branches)[name]) for name, value in original.items())
    with pytest.raises(FileExistsError):
        reference.save_initial_state(branches, path, 'a'*64)
    saved = torch.load(path, weights_only=True)
    del saved['state']['prompt.norm.weight']
    saved['learned_sha256'] = reference.state_digest(saved['state'])
    torch.save(saved, path)
    with pytest.raises(ValueError):
        reference.load_initial_state(branches, path, 'a'*64)


def test_error_context_never_contains_error_message_secrets():
    try:
        raise ValueError('pretend API secret')
    except ValueError as error:
        value = worker.error_context(error, SimpleNamespace(active_phase='train_epoch'), None)
    assert 'pretend API secret' not in json.dumps(value)
    assert value['error_type'] == 'ValueError' and value['frames']


def test_worker_rejects_unapproved_environment_before_imports(monkeypatch,tmp_path):
    for key in ('CL_PROL_EXECUTE','CUDA_VISIBLE_DEVICES','WANDB_MODE'):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match='explicit execution'):
        worker.worker(tmp_path/'absent.json')


def test_exact_counts_old_new():
    result = worker.exact_counts(np.array([0,0,10,10]),np.array([0,1,10,11]),2)
    assert result['seen_correct']==2 and result['old_correct']==result['new_correct']==1


def toy_frozen_hash(branches):
    result = {}
    for branch,module in branches.items():
        state = {name: parameter for name,parameter in module.named_parameters()
            if not reference.learned_name(branch,name) and not name.startswith('head.')}
        result[branch] = {'sha256':reference.state_digest(state),'tensors':len(state)}
    return result


def base_record(side='native'):
    p = {**shared.protocol('validation'),'prompt_layers':[0,1]}
    return {'protocol':p, 'side':side,'mode':'validation','pass':'validation','wandb_url':'synthetic-only',
        'counts':{'observe_calls':0,'optimizer_updates':0,'task_begin_extra_samples':0,'task_end_extra_samples':0},
        'first_batch_audits':[], 'method_audits':[], 'state_audits':[], 'epoch_results':[], 'prediction_files':[],
        'task_results':[], 'routing_audits':[], 'sample_stream_audits':[], 'lr_audits':[],
        'structural_checks':{key:False for key in shared.REQUIRED_CHECKS},
        'common_reference_data_snapshots':{},'common_transform_snapshots':{}}


def synthetic_dataset(args):
    from datasets.utils.continual_dataset import ContinualDataset,MammothDatasetWrapper
    from utils.evaluate import evaluate
    class Raw(torch.utils.data.Dataset):
        def __init__(self,train,task):
            self.data=np.zeros((5000 if train else 1000,8,8,3),dtype=np.uint8)
            self.targets=np.full(len(self.data),10*task,dtype=np.int64)
            self.transform=None
            self.train=train
        def __len__(self):return len(self.data)
        def __getitem__(self,index):
            image=torch.from_numpy(self.data[index].copy()).permute(2,0,1).float()
            return (image,int(self.targets[index]),image) if self.train else (image,int(self.targets[index]))
    class Synthetic(ContinualDataset):
        NAME,SETTING='synthetic-prol-worker','class-il'
        N_CLASSES,N_TASKS,N_CLASSES_PER_TASK,SIZE=100,10,10,(8,8)
        def __init__(self):
            self.args,self.train_loader,self.test_loaders,self.c_task=args,None,[],-1
            self.eval_fn=evaluate
        def get_data_loaders(self):
            task=len(self.test_loaders)
            self.train_loader=DataLoader(MammothDatasetWrapper(Raw(True,task),train=True),batch_size=64,shuffle=True)
            test=DataLoader(MammothDatasetWrapper(Raw(False,task)),batch_size=64)
            self.test_loaders.append(test)
            self.c_task=task
            return self.train_loader,test
        def get_loss(self):return nn.CrossEntropyLoss()
        def log(self,*arguments,**keywords):return None
    return Synthetic(),Raw


def configure_synthetic(monkeypatch,tmp_path,side):
    import main
    from scripts.cl_efficiency import Recorder
    from scripts.ease_reference_utils import dataset_fingerprint
    command=shared.native_command({'mode':'validation'})
    monkeypatch.setattr('sys.argv',['main.py',*command])
    args=main.parse_args(cmd=command)
    args.device,args.num_workers,args.nowand='cpu',0,1
    args.results_path=str(tmp_path/'mammoth')
    dataset,Raw=synthetic_dataset(args)
    record=base_record(side)
    for split,train in [('train',True),('test',False)]:
        raw=Raw(train,0)
        proof=dataset_fingerprint(raw.data,raw.targets,np.arange(len(raw)),raw.transform)
        record['common_reference_data_snapshots']['task1_'+split]=proof
        record['common_transform_snapshots']['task1_'+split]=proof
    recorder=Recorder(tmp_path,gpu='cpu')
    recorder.torch,recorder.device=torch,torch.device('cpu')
    monkeypatch.setattr(worker,'frozen_encoder_hashes',toy_frozen_hash)
    monkeypatch.setattr(worker,'notify_slack',lambda *args,**kwargs:'sent')
    return args,dataset,recorder,record


def test_actual_native_two_task_trainer_resource_instrumentation(monkeypatch,tmp_path):
    from utils import training
    args,dataset,recorder,record=configure_synthetic(monkeypatch,tmp_path,'native')
    torch.manual_seed(1993)
    branches,network=small_native()
    model=ProlPrompt(network,nn.CrossEntropyLoss(),args,None,dataset)
    audit=worker.ProlAudit(branches,dataset,recorder,record,torch,np,model)
    with ExitStack() as stack:
        audit.install_native(stack,training)
        training.train(model,dataset,args)
    audit.close()
    assert record['counts']['observe_calls']==record['counts']['optimizer_updates']==158
    assert recorder.training_examples==10000 and recorder.evaluation_examples==3000 and recorder.tasks_completed==2
    assert len(record['first_batch_audits'])==len(record['state_audits'])==2
    assert model.net.task_updates.tolist()[:2]==[79,79]
    assert len(record['lr_audits'])==158
    assert record['state_audits'][1]['checkpoint_outputs_bitexact']
    assert record['state_audits'][1]['checkpoint_rng_unchanged']
    assert record['state_audits'][1]['checkpoint_dynamic_head_reset_restored']
    assert record['state_audits'][1]['checkpoint_probe_examples']==40
    assert len(record['routing_audits'][1]['batches'])==32
    assert not any(parameter.requires_grad for parameter in model.net.prompt.e_prompt.k_comp_gen.parameters())
    assert len(model.net.prompt.head.heads)==2
    assert len(model.net.query.head.heads)==1
    assert (tmp_path/'task2_state.pt').is_file()


def test_actual_author_two_task_engine_and_native_audits_match(monkeypatch,tmp_path,code):
    from utils import training
    rows=[]
    states=[]
    for side in ['native','author']:
        output=tmp_path/side
        output.mkdir()
        args,dataset,recorder,record=configure_synthetic(monkeypatch,output,side)
        torch.manual_seed(1993)
        if side=='native':
            branches,network=small_native()
            model=ProlPrompt(network,nn.CrossEntropyLoss(),args,None,dataset)
        else:
            branches=small_author(code)
            model=None
        audit=worker.ProlAudit(branches,dataset,recorder,record,torch,np,model)
        with ExitStack() as stack:
            if side=='native':
                audit.install_native(stack,training)
                training.train(model,dataset,args)
            else:
                audit.install_common(stack)
                worker.train_author(code,branches,dataset,audit,record)
        audit.close()
        rows.append(record)
        states.append(reference.learned_state(branches))
    assert rows[0]['first_batch_audits']==rows[1]['first_batch_audits']
    assert rows[0]['lr_audits']==rows[1]['lr_audits']
    assert rows[0]['epoch_results']==rows[1]['epoch_results']
    assert rows[0]['task_results']==rows[1]['task_results']
    assert rows[0]['sample_stream_audits']==rows[1]['sample_stream_audits']
    assert rows[0]['routing_audits']==rows[1]['routing_audits']
    assert all(torch.equal(value,states[1][name]) for name,value in states[0].items())


@pytest.mark.parametrize('side',['native','author'])
def test_compact_clone_restores_disk_learned_values_and_dynamic_heads(monkeypatch,tmp_path,code,side):
    args,dataset,_,_=configure_synthetic(monkeypatch,tmp_path,side)
    torch.manual_seed(1993)
    if side=='native':
        branches,network=small_native()
        model=ProlPrompt(network,nn.CrossEntropyLoss(),args,None,dataset)
        for task in range(2):
            model.meta_begin_task(dataset)
            model.meta_begin_epoch(0,dataset)
            images=torch.randn(3,3,8,8)
            model.meta_observe(images,torch.tensor([0,1,2])+task*10,images,epoch=0)
            model.meta_end_epoch(0,dataset)
            model.meta_end_task(dataset)
    else:
        branches=small_author(code)
        model=None
        source_args=reference.author_args(code,scale_lr=True,batch_size=64,lr=.01,num_tasks=10,epochs=1,print_freq=10000)
        masks=[list(range(task*10,(task+1)*10)) for task in range(10)]
        for task in range(2):
            optimizer,scheduler=reference.begin_author_task(code,branches,source_args,task,masks,task)
            reference.run_author_epoch(code,branches,[(torch.randn(3,3,8,8),torch.tensor([0,1,2])+task*10)],optimizer,
                source_args,task_id=task,class_mask=masks,old_num_k=task)
            scheduler.step()
    payload={'task':2,'state':reference.learned_state(branches),
        'prompt_lifecycle':{name:deepcopy(getattr(branches['prompt'].e_prompt,name)) for name in
            ['old_num_k','new_num_k','old_num_c','new_num_c','kmax_list','lmax_list']}}
    if model is not None:
        payload.update(network_lifecycle={name:value.clone() for name,value in model.net.state_dict().items()
            if name in ['active_task','task_offsets','task_finalized','task_updates']},
            wrapper_extra=model.get_extra_state(),network_extra=model.net.get_extra_state())
    original=branches['prompt'].norm.weight.detach().clone()
    payload['state']['prompt.norm.weight'].add_(.25)
    payload['learned_sha256']=reference.state_digest(payload['state'])
    rng=worker.rng_digest(torch)
    clones,restored=worker.compact_clone(branches,payload,model)
    assert worker.rng_digest(torch)==rng
    assert torch.equal(clones['prompt'].norm.weight,original+.25)
    assert torch.equal(branches['prompt'].norm.weight,original)
    assert len(clones['prompt'].head.heads)==2
    assert clones['prompt'].norm.weight.data_ptr()!=branches['prompt'].norm.weight.data_ptr()
    assert clones['prompt'].blocks[0].norm1.weight.data_ptr()!=branches['prompt'].blocks[0].norm1.weight.data_ptr()
    assert clones['prompt'].blocks[0].attn.qkv.weight.data_ptr()==branches['prompt'].blocks[0].attn.qkv.weight.data_ptr()
    assert not any(parameter.requires_grad for parameter in clones['prompt'].e_prompt.k_comp_gen.parameters())
    assert not any(parameter.requires_grad for parameter in clones['prompt'].head.heads[0].parameters())
    bad=deepcopy(payload)
    del bad['state']['prompt.norm.weight']
    bad['learned_sha256']=reference.state_digest(bad['state'])
    with pytest.raises(ValueError,match='incomplete or corrupt'):
        worker.compact_clone(branches,bad,model)
    if model is not None:
        bad=deepcopy(payload)
        bad['network_lifecycle']['task_updates'][2]=1
        with pytest.raises(ValueError,match='future task state'):
            worker.compact_clone(branches,bad,model)


@pytest.mark.parametrize('task,correct,fails',[(1,899,True),(1,900,False),(2,399,True),(2,400,False)])
def test_absolute_floor_retains_finished_prediction_and_cost_counts(monkeypatch,tmp_path,task,correct,fails):
    branches,_=small_native()
    monkeypatch.setattr(worker,'frozen_encoder_hashes',toy_frozen_hash)
    monkeypatch.setattr(worker,'notify_slack',lambda *args,**kwargs:'sent')
    labels=np.concatenate([np.full(1000,index*10,dtype=np.int64) for index in range(task)])
    predicted=labels.copy()
    predicted[correct:]=(predicted[correct:]+1)%(task*10)
    loaders=[SimpleNamespace(dataset=SimpleNamespace(targets=labels[index*1000:(index+1)*1000],indexes=np.arange(index*1000,(index+1)*1000))) for index in range(task)]
    record=base_record()
    recorder=SimpleNamespace(output=tmp_path,snapshots={},tasks_completed=0)
    audit=worker.ProlAudit(branches,SimpleNamespace(test_loaders=loaders),recorder,record,torch,np)
    audit.task=task-1
    audit.eval_predictions=[predicted]
    audit.eval_batches=([64]*15+[40])*task
    audit.eval_routes=[]
    monkeypatch.setattr(audit,'finalized',lambda:None)
    if fails:
        with pytest.raises(RuntimeError,match='absolute floor'):
            audit.end_evaluation()
        assert recorder.tasks_completed==task
        assert record['accuracy_floor_failure']['seen_accuracy_pct']==100*correct/(task*1000)
    else:
        audit.end_evaluation()
        assert 'accuracy_floor_failure' not in record
    assert record['evaluated_tasks_completed']==task
    assert record['task_results'][-1]['seen_correct']==correct
    assert (tmp_path/f'task{task}_predictions.npz').is_file()
    audit.close()
