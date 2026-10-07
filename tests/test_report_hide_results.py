"""Historical report gates distinguish metadata placement from real changes."""
import copy

import pytest

from scripts.report_hide_results import (PRECISION_FIELDS, compatible_history,
                                        markdown, method_summary)


@pytest.fixture
def history():
    precision = {'parameter_dtype': 'float32', 'autocast': False,
                 'matmul_precision': 'highest', 'tf32_allowed': False,
                 'cudnn_tf32_allowed': True, 'cudnn_deterministic': True,
                 'cudnn_benchmark': False}
    target = {'checkpoint_sha256': 'checkpoint-fixture', 'gpu': '1',
              'precision': {**precision, 'torch_cpu_threads': 2},
              'data_snapshots': {}}
    for split in ('train', 'test'):
        target['data_snapshots']['task1_' + split] = {
            'n_samples': 5000 if split == 'train' else 1000,
            'class_counts': {'0': 500 if split == 'train' else 100},
            'samples_ordered_sha256': split + '-ordered-fixture',
            'samples_unordered_sha256': split + '-unordered-fixture',
            'global_ids_ordered_sha256': split + '-ids-fixture',
            'transform_repr_sha256': split + '-transform-fixture'}
    record = {'status': 'completed', 'pass': 'measurement',
              'metrics': {'evaluation_examples': 1000},
              'precision': precision, 'torch_cpu_threads': 2,
              'data_snapshots': copy.deepcopy(target['data_snapshots'])}
    records = [copy.deepcopy(record) for _ in range(3)]
    document = {'status': 'completed', 'pilot_status': 'pass',
                'checkpoint_sha256': target['checkpoint_sha256'],
                'protocol': {'dataset': 'seq-cifar100-224', 'dataset_config': 'ease',
                             'gpu': '1', 'efficiency': {'tasks': 1, 'batch_size': 64,
                                                      'seed': 1993, 'num_workers': 4}}}
    return document, records, target


def test_field_location_difference_and_unrelated_metadata_preserve_comparison(history):
    document, records, target = history
    records[0]['precision']['implementation_note'] = 'Extra observational metadata'
    result = compatible_history(document, records, target)
    assert result['status'] == 'compatible'
    assert result['required_precision_fields'] == list(PRECISION_FIELDS)
    evidence = result['cpu_thread_evidence']
    assert evidence['status'] == 'matched'
    assert evidence['target']['sources'] == {'precision': 2}
    assert all(entry['sources'] == {'record': 2} for entry in evidence['historical'])


@pytest.mark.parametrize('field', PRECISION_FIELDS)
def test_each_real_precision_difference_blocks_comparison(history, field):
    document, records, target = history
    original = records[1]['precision'][field]
    records[1]['precision'][field] = not original if type(original) is bool else 'different-policy'
    result = compatible_history(document, records, target)
    assert result['status'] == 'excluded'
    assert any('precision policy ' + field in reason for reason in result['reasons'])


@pytest.mark.parametrize('missing', ['omitted', 'null'])
def test_required_precision_evidence_cannot_be_missing_in_both_sides(history, missing):
    document, records, target = history
    for precision in [target['precision'], *(record['precision'] for record in records)]:
        if missing == 'omitted':
            del precision['autocast']
        else:
            precision['autocast'] = None
    result = compatible_history(document, records, target)
    assert result['status'] == 'excluded'
    assert 'precision policy autocast differs or lacks required evidence' in result['reasons']


def test_false_precision_value_is_not_equivalent_to_integer_zero(history):
    document, records, target = history
    records[0]['precision']['autocast'] = 0
    assert compatible_history(document, records, target)['status'] == 'excluded'


def test_missing_ease_thread_metadata_stays_unknown_without_exclusion(history):
    document, records, target = history
    for record in records:
        del record['torch_cpu_threads']
    result = compatible_history(document, records, target)
    assert result['status'] == 'compatible'
    evidence = result['cpu_thread_evidence']
    assert evidence['status'] == 'metadata_missing'
    assert all(entry == {'value': None, 'sources': {}, 'status': 'not_recorded'}
               for entry in evidence['historical'])


@pytest.mark.parametrize('location', ['record', 'precision'])
def test_known_thread_count_difference_blocks_comparison(history, location):
    document, records, target = history
    if location == 'record':
        records[1]['torch_cpu_threads'] = 4
    else:
        del records[1]['torch_cpu_threads']
        records[1]['precision']['torch_cpu_threads'] = 4
    result = compatible_history(document, records, target)
    assert result['status'] == 'excluded'
    assert 'recorded CPU thread count differs' in result['reasons']


def test_conflicting_thread_locations_cannot_be_silently_normalized(history):
    document, records, target = history
    records[1]['precision']['torch_cpu_threads'] = 4
    result = compatible_history(document, records, target)
    assert result['status'] == 'excluded'
    assert result['cpu_thread_evidence']['historical'][1]['status'] == 'invalid'


def test_ranpac_training_transform_difference_remains_excluded(history):
    document, records, target = history
    for record in records:
        record['data_snapshots']['task1_train']['transform_repr_sha256'] = 'ranpac-transform-fixture'
    result = compatible_history(document, records, target)
    assert result['status'] == 'excluded'
    assert result['cpu_thread_evidence']['status'] == 'matched'
    assert result['reasons'] == ['historical train transform_repr_sha256 differs or lacks evidence']


def test_markdown_names_only_reused_methods_and_discloses_missing_threads_and_exclusions(history):
    document, records, target = history
    for record in records:
        del record['torch_cpu_threads']
    ease = method_summary('ease', records, {}, historical=True)
    ease['compatibility'] = compatible_history(document, records, target)
    ranpac_records = copy.deepcopy(records)
    ranpac_records[0]['data_snapshots']['task1_train']['transform_repr_sha256'] = 'different-transform'
    exclusion = {'model': 'ranpac', **compatible_history(document, ranpac_records, target)}
    report = {'validation': {'acceptance': {'status': 'pass'},
                            'first_batch_comparison': {'status': 'pass'},
                            'accuracy_metrics': [], 'reference': {}, 'native': {}},
              'efficiency': {'methods': [ease, method_summary('hide-prompt', records, {})],
                             'historical_exclusions': [exclusion]}}
    rendered = markdown(report)
    assert '기존 측정 재사용: **EASE**' in rendered
    assert '기존 측정 미기록; 일치 여부 미확인, 값 추정 없음' in rendered
    assert '| RanPAC | historical train transform_repr_sha256' in rendered
    assert '기존 EASE/RanPAC/L2P/DualPrompt는 원본 측정을 재사용합니다' not in rendered
