# PROL 공식 활성 코드의 Mammoth 이식

Base 구현·독립 저자 코드와의 CPU 대조를 마쳤으며, 사용자가 후속 GPU 검증·조건부 효율성 측정을 승인했습니다. 승인한 범위는 [run_plan.json](run_plan.json), 이번 실행·검사 기록은 [RUN_VALIDATION.md](RUN_VALIDATION.md)에 기록합니다. 초기 CPU 검사 기록은 [CPU_VALIDATION.md](CPU_VALIDATION.md), 가중치 점검은 [cpu_weights.json](cpu_weights.json)에 보관합니다. CPU 검사는 실제 데이터 정확도나 효율성 측정 결과가 아닙니다.

구현 기준은 [공식 저장소](https://github.com/anwarmaxsum/PROL)의 commit `bfff8418a4f603a24ae578f1e108bfac89af1e18`입니다. 전체 tracked 파일 22개의 SHA-256과 활성 경로를 [source_review.json](source_review.json)에 고정했습니다. 소스는 `/tmp/cl-prol-official-20261008`에 보관하며, 비교 도구는 파일·commit 변경을 거절합니다. 논문식 수정은 이번에 포함하지 않습니다.

## 특징과 실제 코드 동작

PROL은 별도 query ViT의 특징으로 경량 K/V prompt를 생성하고, class key와 scaler/shifter로 이를 조절합니다. 기본 prefix 길이는 5이고 layer 0–6에 적용합니다. 각 layer는 현재 생성 prompt와 바로 이전 layer의 생성 prompt를 합합니다. 첫 태스크 뒤 generator를 동결하고, 새 classifier head를 추가하며 이전 head를 동결합니다. 원본 예제 replay는 사용하지 않습니다. [고정한 prompt 코드](https://github.com/anwarmaxsum/PROL/blob/bfff8418a4f603a24ae578f1e108bfac89af1e18/promptL.py), [ViT](https://github.com/anwarmaxsum/PROL/blob/bfff8418a4f603a24ae578f1e108bfac89af1e18/vision_transformerL.py)

장점은 1epoch 온라인 학습과 작은 generator를 직접 비교할 수 있다는 점입니다. 단, 작은 prompt parameter 수만으로 실행 비용이 작다고 결론낼 수 없습니다. query와 prompted ViT의 두 forward, trainable LayerNorm을 포함한 backward, 768×768 generalization matrix, 태스크별 classifier 증가를 모두 계측해야 합니다.

| 공식 활성 경로에서 확인한 항목 | 이번 이식·검증 기준 |
|---|---|
| Generalization loss | `GM = normalize(query).T @ normalize(prompted) / B`. diagonal 보존 항과 `GM - GM @ I` 항을 그대로 계산합니다. 뒤 항은 수학적으로 0이며 논문의 off-diagonal 보존과 다릅니다. 임의 수정하지 않습니다. |
| Trainability | query는 모두 동결합니다. prompted encoder의 patch/QKV/MLP/token은 동결하지만 block norm1/norm2와 final norm은 학습합니다. 새 head, class key, scaler/shifter도 학습합니다. |
| Generator | 첫 태스크에서 학습하고 두 번째 태스크 시작에 동결합니다. class key는 하나의 parameter이며 old-key orthogonality 경로에 detach가 없으므로 과거 row의 변화도 허용합니다. |
| Similarity assignment | `similarity_new[:, idx[:, c]] = similarity_top_k[:, c]`는 배치의 여러 행을 함께 갱신합니다. row-wise scatter로 고치지 않습니다. top_k_l=1이면 generator는 column0만 사용하여 배치 구성에 따른 결과 변화가 생깁니다. |
| Class indexing | training의 key 후보·scaler index는 현재 클래스 구간의 local index이며, matched-label similarity는 global label을 사용합니다. offset 보정으로 바꾸지 않습니다. |
| Clamp | scaler는 0.999–1.001, shifter는 ±0.0001로 forward 중 clamp됩니다. **평가 중에도 상태가 바뀔 수 있습니다.** checkpoint 대조에서는 평가·clamp 순서를 일치시킵니다. |
| Loss | `current CE + .001 * seen CE + .1 * generalization - reduce_sim + reduce_sim2`. 마지막 prompt layer의 항을 사용합니다. task0에서는 reduce_sim2가 없습니다. |
| Hard/soft transition | 두 개의 별도 알고리즘이 아닙니다. CE 조합이 0.3 미만이면 epoch의 나머지 batch마다 cosine scheduler를 step합니다. generalization/pull term을 더하기 전의 CE 조합을 기준으로 합니다. |
| LR schedule | inner cosine T_max=20·eta_min=0.0005와 epoch-end cosine T_max=1·eta_min=0을 모두 유지합니다. inner scheduler는 20step 이후에도 진행해 LR이 다시 올라갈 수 있습니다. task2 norm 시작 LR은 eta_min보다 작아져 norm LR도 증가할 수 있습니다. 모든 group의 실제 LR trace가 필요합니다. |
| Query mode·head | 소스 query는 no_grad지만 eval() 전환은 하지 않습니다. dropout/drop-path는 0입니다. 사용하지 않는 query classifier도 기본 num_tasks=5에 따른 20class head를 유지하여 parameter·메모리 집계에 포함합니다. |

[공식 main](https://github.com/anwarmaxsum/PROL/blob/bfff8418a4f603a24ae578f1e108bfac89af1e18/main_prol.py), [활성 학습 engine](https://github.com/anwarmaxsum/PROL/blob/bfff8418a4f603a24ae578f1e108bfac89af1e18/engine_prol.py), [classifier](https://github.com/anwarmaxsum/PROL/blob/bfff8418a4f603a24ae578f1e108bfac89af1e18/linears.py)에 근거합니다. 위 동작을 보존한 구현은 `official-active-path`이며 논문의 전체 수학적 정의나 발표 정확도 재현을 의미하지 않습니다.

## 승인한 실행 조건

[validation_plan.json](validation_plan.json)과 [execution_plan.json](execution_plan.json)은 초기 CPU 준비 단계의 미실행 제안 기록입니다. 이번 승인 범위는 별도의 [run_plan.json](run_plan.json)에 고정합니다. 준비 도구에는 실행 옵션이 없으며 credentials를 읽거나 W&B·Slack에 쓰지 않습니다. 실제 worker·suite·publication은 명시적인 `--execute`로 실행합니다.

| 항목 | 승인 조건 |
|---|---|
| 데이터 | seq-cifar100-224/ease, 원래 class0–99, 10tasks×10classes, task별 train5000/test1000 |
| 가중치 | 공통 로컬 AugReg ViT-B/16 IN21K→IN1K fine-tuned NPZ. 양쪽 encoder의 초기 canonical150 tensors hash 대조 |
| Training | batch64, seed1993, workers4, FP32, CPUthreads2, 물리 GPU1 |
| LR | source config 기본 nominal0.01 → initial effective0.0025. README의 nominal0.05→0.0125는 대안이며 추가 실험으로 준비하지 않음 |
| Source/native 검증 | 각 첫2tasks×1epoch 1회. task1/seen2/old2/new2 accuracy, gradient·state·입력·LR·count·checkpoint 대조 |
| 효율성 | 구현 검증과 성능 정상성 점검 통과 후 native 첫task×1epoch 3회. 별도 FLOPs1회. 총 GPU 최대6회 |
| Accuracy gate | native 감소≤1.5%p 통과, >1.5–3%p 점검, >3%p 실패. 비정상적으로 낮은 성능은 양쪽이 같아도 실패 처리 |
| Absolute floor | task1<90%면 해당 worker에서 즉시 중단. task2 seen<20%도 중단. 20%는 붕괴 방지선이며 적정 성능 기준이 아니므로 old/new 정확도·loss·LR 추세를 별도 점검하고 근거를 저장한 뒤 pilot을 진행 |
| 저장·알림 | local + laymond1/cl-efficiency, 기존 Slack webhook. weights는 local에 두고 metric/config/원자료/검증 evidence만 업로드 |

instrumented author/native worker·suite·report·publication을 CPU 검사한 뒤 GPU에서 검증합니다. `--through validation_native`로 첫 두 실행을 마친 후 입력·gradient·정수 기반 정확도를 비교하고, 정상성 검토 receipt를 저장해야 효율성 실행을 재개할 수 있습니다. 원형 검증과 pilot은 같은 1epoch recipe이며 별도 단축 검증을 반복하지 않습니다. 실패한 실행·디렉터리를 자동 재시도하거나 덮어쓰지 않으며 LR·seed도 자동 변경하지 않습니다. 중단 시 관측된 원자료를 부분 보고서로 저장·게시하고 효율성 평균은 만들지 않습니다.

모든 worktree는 data `/mnt/sdab1/share/data`, cache `/mnt/sdab1/userHome/wslim/.cache`와 기존 torch/HF/CLIP 하위 경로를 공유합니다. 로컬 NPZ만 로드하며 가중치·데이터를 내려받지 않습니다. CLI의 `--nowand 1` 명령은 training core 점검용이며 최종 계측 worker가 online W&B와 Slack을 관리합니다. 명령만 실행하면 예정한 효율성 자료가 자동 생성된다는 의미가 아닙니다.

시간은 training phase와 전체 process를 구분합니다. CUDA allocated/reserved peak, process-tree RSS·CPU time, raw power CSV와 GPU board energy를 기록하도록 준비합니다. FLOPs는 지원 연산만 별도 실행하여 합산하고 측정3회 평균에 섞지 않습니다. GPU board energy는 CPU/RAM 또는 시스템 전체 에너지가 아니며, process-tree RSS는 공유 page를 중복 합산할 수 있습니다. 효율성 반복은 같은 seed의 실행 비용 변동이며 다중 seed 정확도 추정이 아닙니다.

## 범위

Mammoth `train`과 `meta_begin_task/meta_observe/meta_end_task`를 유지합니다. 공식 `.cuda()` head 생성은 현재 device로 옮기는 interface 적응만 적용합니다. task oracle·future-class 평가·replay 없이 seen-class class-IL을 제공합니다. framework stream resume/loadcheck는 이번 단계의 지원 범위에 포함하지 않습니다. 명시적인 state_dict 대조는 dynamic head·freeze·task state를 포함해 검증합니다.

전체10tasks, Small/Tiny training, 새 seed, 예산 비교, 방법 간 비교 확대, 논문식 bug 수정, remote git push는 후속 범위입니다.
