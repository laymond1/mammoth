# 추가 구현할 CL 방법과 계산 예산 비교 후보

이 문서는 2026년 10월 1일 조사와 10월 6일 논문·저자 코드 검토 당시의 **14개 항목: 방법 후보 13개와 BudgetCL 평가 프로토콜·기준선 1개**를 보존합니다. 구현 순위와 이식 난이도는 당시 검토 의견이며, 다른 논문의 정확도를 이 워크스페이스의 예상 결과로 사용하지 않습니다.

2026-10-08 현재 EASE·HiDe-Prompt와 EWC-LoRA·SD-LoRA·InfLoRA·CL-LoRA, RainbowPrompt와 PROL이 구현되어 소스 엔트리는 **78개**입니다. CL-LoRA는 초기 14개 검토 이후 사용자 결정으로 추가했습니다. 네 LoRA 방법은 고정한 공통 저자 저장소 변형의 Base 첫2태스크 구현 검증을 통과했고, RainbowPrompt는 learned gate가 없는 공식 활성 경로의 CPU172검사·첫2태스크 및 단축 GPU 검증·효율성 pilot을 완료했습니다. [현재 알고리즘 목록](algorithms.md), [LoRA 검증 결과](low_rank/RESULTS.md), [RainbowPrompt 결과](rainbow_validation/RESULTS.md), [방법별 실행 이력](README.md)을 참고합니다. **PROL은 공식 활성 코드 이식·CPU 대조 후 GPU checkpoint 검증에서 실패했고, 추론 진단의 반복 출력 변동을 확인한 뒤 사용자 결정으로 추가 작업을 보류하고 이번 비교에서 제외했습니다.** [PROL 실패·진단과 보류 결정](prol_diagnostic/README.md)을 보존합니다. 아래 후보 표는 당시 논문의 특징·이식 제안을 보존하며 실제 구현·측정 완료 범위는 최신 문서로 확인합니다.

우선 **EWC-LoRA, SD-LoRA, InfLoRA, PROL, RainbowPrompt, ProbPrompt, EASE, HiDe-Prompt**의 8개 방법을 함께 검토합니다. 기존 `second_order`의 LoRA와 Fisher 코드, L2P·DualPrompt·CODA-Prompt, RanPAC 구현을 참고할 수 있고, 정규화·고정 용량·prompt 생성·prompt 다양성·확장형 adapter·task routing이라는 서로 다른 설계 선택을 비교할 수 있습니다. BudgetCL과 aL-SAR은 아래의 계산 예산 비교 항목으로 추가했습니다.

## 우선 검토할 후보

| 방법과 발표 | 핵심 특징 | 장점 | 단점과 확인할 비용 | 이식 방향과 검토 의견 |
|---|---|---|---|---|
| **EWC-LoRA**, ICLR 2026 | 공유 low-rank update를 EWC로 정규화하고 full-dimensional 중요도를 low-rank 표현으로 추정 | 태스크별 adapter pool 증가를 피하고 adapter 저장·추론 비용을 일정하게 유지하는 설계 | Fisher 계산과 정규화 강도에 따른 적응성 저하를 측정해야 함. 분류기까지 포함한 전체 시스템 크기가 일정하다는 의미는 아님 | **우선순위 1**. `second_order`·`ewc_on`·LoRA tuner와 비교하되 논문의 중요도 추정식을 별도로 구현. [논문](https://arxiv.org/abs/2602.17559), [저자 코드](https://github.com/yaoyz96/low-rank-cl) |
| **SD-LoRA**, ICLR 2025 Oral | LoRA의 방향과 크기 학습을 분리해 순차 적응 | 선택용 별도 routing 없이 end-to-end 학습과 효율적인 추론을 목표로 함 | magnitude/direction의 학습 스케줄과 구현 변형에 민감할 수 있음. adapter와 누적 업데이트의 실제 저장 방식 확인 필요 | **우선순위 1**. 기존 LoRA tuner와 연결; `second_order --tuning_style lora`를 현재 코드 기준선으로 사용. [논문](https://openreview.net/forum?id=5U1rlpX68A), [저자 코드](https://github.com/WuYichen-97/SD-Lora-CL) |
| **InfLoRA**, CVPR 2024 | 현재 입력과 과거 태스크의 subspace를 이용해 LoRA update 간섭을 제어 | 원본 replay 없이 명시적인 간섭 억제; 최근 LoRA CL과 비교할 안정적인 기준점 | SVD/subspace 수집과 basis 저장 비용. 제약이 강해지면 새 태스크의 학습 자유도가 줄 수 있음 | **우선순위 1의 비교 기준선**. 최신 발표는 아니지만 LoRA 계열 비교에 필요. Q/V LoRA와 task-end subspace 코드를 구현. [논문](https://arxiv.org/abs/2404.00228), [저자 코드](https://github.com/liangyanshuo/InfLoRA) |
| **PROL**, ICCV 2025 | 단일 경량 prompt generator, scaler/shifter, 사전학습 지식 보존, hard/soft update | 온라인 스트림과 1회 데이터 접근 조건에 맞고 prompt pool 확장 비용을 줄이려는 설계 | 공식 코드에는 학습 가능한 LayerNorm·배치 간 indexing·평가 clamp·0이 되는 비대각 항이 있음; 두 ViT 및 generalization matrix 비용 확인 필요 | **사용자 결정으로 보류·이번 비교 제외**. CPU 이식 대조 완료, 저자 첫태스크 checkpoint 검증 실패 후 별도40회 추론에서 반복 출력 변동 확인. 추가 진단·교정·재학습은 중단. [진단과 보류 기록](prol_diagnostic/README.md), [논문](https://arxiv.org/abs/2507.12305), [저자 코드](https://github.com/anwarmaxsum/PROL) |
| **RainbowPrompt**, ICCV 2025 | 기존/새 base prompt를 변환·정렬해 통합하고 확률적 gate로 활성 layer 선택 | prompt 다양성과 태스크 간 지식 결합을 함께 다룸 | prompt 변환과 alignment/gate 비용. gate가 sparse하다는 사실만으로 실제 GPU FLOPs/에너지 절감이 보장되지는 않음 | **우선순위 1**. 저자 코드가 DualPrompt 기반이어서 현재 구현과 비교·이식하기 유리하다고 판단. [논문](https://arxiv.org/abs/2507.22553), [저자 코드](https://github.com/Kiseong0753/RainbowPrompt) |
| **ProbPrompt**, ECCV 2026 | prompt를 확률 분포의 mixture로 표현하고 sampling과 distribution regularization으로 collapse 완화 | 최근 prompt 다양성 문제를 직접 겨냥함. 논문은 샘플을 prompt 적용 전에 집계하여 입력당 여러 backbone forward를 피함 | 분산·mixture·sampling 수의 선택과 확률성 관리. 추가 통계 저장 및 정규화 비용을 실제 측정해야 함 | **우선순위 1**. CODA-Prompt/L2P와 비교할 최신 후보. seed별 분산도 후속 평가. [논문](https://arxiv.org/abs/2607.04711), [저자 코드](https://github.com/cvlab-yonsei/ProbPrompt) |
| **EASE**, CVPR 2024 | frozen ViT에 태스크별 FFN adapter를 추가하고 feature를 결합. 새 subspace에서 과거 클래스 prototype을 보완 | 원본 replay 없이 과거 adapter를 보존하며 새 태스크에 적응. 확장형 PEFT의 비교 기준으로 유용 | adapter·분류기 저장량 증가. 공유 backbone이어도 adapter마다 transformer를 실행하므로 task-end와 추론 비용이 태스크 수에 따라 증가 | **첫 구현 완료**. [구현·검증 결과](ease/README.md), 첫2태스크 검증·Base64 효율성 측정·로컬 병합 완료. 전체 CL 재현과 Tiny·Small은 후속 단계. [논문](https://arxiv.org/abs/2403.12030), [저자 코드](https://github.com/sun-hailong/CVPR24-Ease) |
| **HiDe-Prompt**, NeurIPS 2023 Spotlight | within-task prediction, task-identity inference(TII), task-adaptive prediction(TAP)을 분리. task별 prompt와 feature 분포 통계 사용 | prompt 표현과 task 선택을 함께 개선하고 routing 실패를 분해해 분석할 수 있음 | frozen ViT+MLP/head의 **태스크별 TII 학습**, prompt 학습·통계·alignment 비용. 추론에는 TII용 및 선택 prompt용 ViT forward가 필요하고 routing 오류가 최종 성능을 제한 | **우선순위 1로 추가**. DualPrompt와 SLCA의 통계 코드를 참고하되 두 학습 단계와 task routing을 구현. 이식 난이도 중상. [논문](https://arxiv.org/abs/2310.07234), [저자 코드](https://github.com/thu-ml/HiDe-Prompt) |

## 다음으로 검토할 후보

| 방법과 발표 | 핵심 특징과 장점 | 단점과 확인할 비용 | 추가할 이유와 검토 의견 |
|---|---|---|---|
| **DGS**, CVPR 2026 | LoRA gradient의 dual update와 semantic shift 기반 classifier alignment, patch-level alignment로 안정성/적응성 균형 제어 | projection·prototype 통계·patch loss가 추가되어 작은 학습 파라미터 수와 실행 비용이 다를 수 있음 | 최신 gradient 제어와 기존 `second_order` 비교. 이식 난이도는 중상으로 판단. [논문](https://openaccess.thecvf.com/content/CVPR2026/html/Li_DGS_Dual_Gradient_and_Semantic-Shift_Guided_Low-Rank_Adaptation_for_Class_CVPR_2026_paper.html), [저자 공개 코드](https://github.com/ParadeLee/DGS) |
| **PEARL의 dynamic LoRA rank**, TMLR 2026, arXiv 최초 2025 | reference task weight와의 거리로 태스크/층별 LoRA rank를 배분. ViT 외 CNN도 다룸 | reference training와 task proximity 추정 비용; 태스크별 rank를 조절해도 장기 저장량이 반드시 고정되는 것은 아님 | rank 대 정확도·저장량의 관계 연구에 적합. 공식 코드 위치는 이번 조사에서 확인하지 못함. [논문](https://arxiv.org/abs/2505.11998) |
| **LW2G**, 저자 저장소 표기 IJCV 2026 | 태스크 간 차이에 따라 prompt set을 공유하거나 새로 생성; HFC 및 gradient constraint 사용 | grow 여부 평가와 subspace 계산 비용. HiDe-Prompt 조합은 태스크별 TII 학습 단계도 포함해야 함 | 장기 prompt pool 증가량의 비교에 적합. 저자 코드는 DualPrompt에도 직접 조합 가능하다고 설명. [저자 코드와 방법 설명](https://github.com/RAIAN08/LW2G) |
| **Prompt sharing과 history-aware modulation**, CVPR 2026의 “Is Parameter Isolation Better for Prompt-Based Continual Learning?” | global pool을 공유하고 task-aware sparse routing 및 과거 활성 통계로 prompt update를 조절 | routing·활성 통계 관리 비용과 공유 prompt의 간섭; sparse 활성의 실제 실행 효율을 확인해야 함 | 고정 task별 prompt 할당을 대체하는 최신 설계. 논문에 code/model 공개 예정으로 적혀 있으며 공식 구현은 이번 조사에서 확인하지 못함. [논문](https://arxiv.org/abs/2601.20894), [CVPR 원문](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_Is_Parameter_Isolation_Better_for_Prompt-Based_Continual_Learning_CVPR_2026_paper.pdf) |

PEARL에는 이름이 같은 별도 **AAAI 2025 prompt 방법**도 있습니다. 위 추천의 PEARL은 arXiv `2505.11998`의 dynamic rank 방법을 의미합니다. 두 방법의 코드나 citation을 섞지 않아야 합니다.

## 계산 예산 비교에 추가할 항목

| 항목과 발표 | 특징과 장점 | 단점과 확인할 비용 | 추가 가능성과 권장 범위 |
|---|---|---|---|
| **BudgetCL**, CVPR 2023 | 시간 단계마다 새 데이터가 도착하고, 관측한 전체 이력을 보관하면서 단계별 학습 계산량을 제한하는 평가 설정. uniform-history sampling+CE 기준선과 sampling·distillation·FC correction 등을 비교 | 원본 데이터의 전체 이력 저장이 가능하므로 고정 replay 메모리 방법과 조건이 다름. 논문의 비용에 따른 update 수 배분을 현재 profiler 측정값과 구분해야 함 | **추가 가능**. 단일 `budgetcl` 알고리즘보다 **예산 프로토콜 + uniform-history 기준선**으로 추가하는 것이 적절. 난이도 중. [논문](https://arxiv.org/abs/2303.11165), [저자 코드](https://github.com/drimpossible/BudgetCL) |
| **aL-SAR**, ICLR 2025 Spotlight | Fisher 정보와 계산 비용으로 앞쪽 layer prefix를 적응적으로 동결(aL). 사용 빈도와 클래스 gradient 유사도를 함께 반영하는 replay 선택(SAR). 고정 replay 메모리에서 더 유용한 update에 계산을 배분 | Fisher·per-example gradient·샘플링 통계 비용이 추가됨. 동결 파라미터 수만으로 실제 backward 절감을 입증할 수 없고, 현재 저자 CIL 코드의 ViT 경로는 similarity retrieval을 생략 | **추가 가능**. 우선 ResNet의 native Mammoth 구현을 권장. 난이도 중상; ViT/LoRA 확장은 별도 검토. [논문](https://arxiv.org/abs/2410.15143), [저자 코드](https://github.com/snumprlab/budgeted-cl) |

위 표의 aL-SAR 장점과 비용을 구분하면, 목표는 **계산을 줄여 같은 예산에서 더 많은 유효 학습을 수행하는 것**이고, 실제 절감 여부는 추가 통계 연산을 포함해 측정해야 합니다. [BudgetCL·aL-SAR의 이식 범위, 소스 점검과 비교 조건](budgeted_cl_review.md)에 상세 검토를 기록했습니다. 두 항목 모두 아직 실행 manifest에 넣지 않았습니다.

## EASE와 HiDe-Prompt 이식 시 포함할 비용

**EASE**는 RanPAC의 adapter 구조를 참고할 수 있지만, 첫 태스크 adapter만 학습하는 RanPAC과 달리 adapter를 계속 누적합니다. 기존 adapter를 `ModuleList`로 등록해 checkpoint·device 이동·파라미터 집계에 포함해야 합니다. Mammoth는 task-end hook 이후 평가하므로 새 adapter 생성은 다음 `begin_task`에 두고, 현재 `end_task`에서 prototype 계산·보완을 마치는 방향을 권장합니다. 저자 코드의 train→eval→`after_task` 순서를 그대로 `end_task`로 옮기면 평가 feature 수와 분류기 차원이 어긋날 수 있습니다. [EASE의 ViT 구현](https://github.com/sun-hailong/CVPR24-Ease/blob/6d67508ed8b7c4d23350ed4eb28c123914b40ae3/backbone/vit_ease.py), [학습·평가 순서](https://github.com/sun-hailong/CVPR24-Ease/blob/6d67508ed8b7c4d23350ed4eb28c123914b40ae3/trainer.py)

**HiDe-Prompt**의 TII는 한 번만 수행하는 사전학습이 아닙니다. 저자 CIFAR-100 스크립트는 TII 20 epochs/task와 prompt 50 epochs/task를 별도로 실행하며, 후속 태스크에는 alignment 비용도 있습니다. 이식 시 해당 태스크까지 학습한 TII checkpoint와 동일한 class/task mapping을 사용해야 합니다. 마지막 태스크의 TII checkpoint를 초기 태스크에 쓰면 미래 데이터 정보가 섞입니다. TII와 prompt 단계의 시간·메모리·에너지를 모두 수집하고, 제공받은 task ID로 routing을 대체하지 않는 방향을 권장합니다. [공식 실행 스크립트](https://github.com/thu-ml/HiDe-Prompt/blob/0ed433334ff2cfb206c5e2899651c3044ec85f92/training_scripts/train_cifar100_vit.sh), [TII 학습](https://github.com/thu-ml/HiDe-Prompt/blob/0ed433334ff2cfb206c5e2899651c3044ec85f92/engines/hide_tii_engine.py)

## 현재 코드와 연결해 검토할 실험

첫 후보 묶음은 기존 L2P·DualPrompt·CODA-Prompt·DAP·STARPrompt에 PROL·RainbowPrompt·ProbPrompt·HiDe-Prompt를 더한 prompt 비교입니다. 두 번째 묶음은 기존 `second_order`의 LoRA/IA3 설정 및 선택적 LoRA를 지원하는 TAK에 InfLoRA·SD-LoRA·EWC-LoRA를 더한 비교입니다. EASE는 RanPAC과 함께 확장형 adapter와 단일 adapter의 저장량·추론 비용을 비교합니다. `second_order`와 TAK는 서로 다른 backbone·사전학습·정규화 구현이므로 이름만으로 동일한 LoRA 기준선으로 취급하지 않습니다.

BudgetCL과 aL-SAR을 추가하려면 자원 측정 외에 **실제 계산 예산을 제한하는 실행 제어**가 필요합니다. 현재 도구는 첫 태스크 자원 pilot이며, 공통 `1 epoch` 또는 `n_iters`가 같은 FLOPs 예산을 의미하지 않습니다. 관측한 데이터 범위·replay 정책·backbone을 맞춘 예산 비교를 별도 protocol로 준비해야 합니다.

학습 파라미터가 적어도 prompt gradient를 계산하려면 frozen backbone의 여러 층을 역전파할 수 있고, query용 별도 backbone, teacher, feature 분포, Fisher, adapter pool도 메모리를 차지합니다. 따라서 trainable parameter 수와 함께 **전체 상주 메모리, 태스크 준비/마무리 비용, 실제 처리량, GPU 에너지**를 봐야 합니다.

첫 태스크 pilot은 망각 억제나 과거 task와의 간섭을 거의 검증하지 못합니다. EASE의 누적 adapter·과거 prototype 보완, HiDe-Prompt의 여러 task 간 routing·과거 통계 효과도 검증할 수 없습니다. aL-SAR의 동결·샘플링 비용은 첫 태스크에서도 조사할 수 있지만 warm-up 이후 충분한 update가 있는지 확인해야 합니다. 후보를 추린 뒤 3 seeds와 전체 task sequence로 정확도·평균 incremental accuracy·forgetting·BWT 및 task별 저장량 증가를 다시 확인하는 것을 추천합니다. 이 후속 실험은 아직 실행안에 포함해 시작하지 않았습니다.
