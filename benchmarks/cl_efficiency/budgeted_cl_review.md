# BudgetCL과 aL-SAR의 Mammoth 추가 가능성

2026년 10월 6일 검토 결과, **두 항목 모두 추가할 수 있습니다.** BudgetCL은 예산을 제한하는 평가 프로토콜과 기준선으로, aL-SAR은 native CL 모델로 이식하는 방향을 권장합니다. 이식 난이도와 구현 순서는 현재 코드와 비교한 검토 의견입니다. 모델 구현·학습·추론·실제 알림 전송은 수행하지 않았습니다.

| 구분 | BudgetCL | aL-SAR |
|---|---|---|
| 발표·성격 | CVPR 2023, 계산 예산 아래 여러 학습 구성 요소를 비교하는 문제 설정 | ICLR 2025 Spotlight, adaptive layer freezing와 Similarity-Aware Retrieval을 결합한 알고리즘 |
| 데이터 접근 | 단계마다 새 chunk 도착, 관측한 과거와 현재 데이터 전체를 보관 가능 | 온라인 샘플 도착, 고정 크기 replay memory와 update credit 사용 |
| 주요 비교/기능 | uniform-history sampling+CE, sampling·distillation·FC correction·FC-only 학습 | Fisher 정보/연산 비용으로 앞쪽 layer prefix 동결, 사용 빈도와 class gradient 유사도로 replay 선택 |
| 장점 | 논문이 근사한 teacher·후보 평가 forward 비용을 반영해 단순 기준선과 같은 예산으로 비교 | 학습에 덜 필요한 backward와 반복 replay에 쓰는 예산을 다른 update로 배분 |
| 부담·제한 | 전체 이력 저장 비용. 구성 요소마다 가능한 update 수가 다름 | Fisher·per-example gradient·통계 저장 비용. 동결이 실제 kernel 비용을 줄이는지 검증 필요 |
| 권장 추가 형태 | 예산 프로토콜 + 명시적으로 이름 붙인 uniform-history 기준선 | 우선 ResNet의 `alsar` 모델; ViT/LoRA는 후속 확장 |
| 난이도 판단 | 중: 데이터 이력과 예산 제어 필요 | 중상: 온라인 상태·동결·gradient 통계·비용 추정의 일관성 필요 |

문제 설정과 발표 정보는 [BudgetCL 논문](https://arxiv.org/abs/2303.11165), [aL-SAR 논문](https://arxiv.org/abs/2410.15143) 및 [저자 aL-SAR 저장소](https://github.com/snumprlab/budgeted-cl)에 따릅니다. aL-SAR의 SAR에는 빈도 기반 선택과 유사도 보정이 함께 들어갑니다.

## BudgetCL: 프로토콜과 기준선을 먼저 추가

원논문은 관측한 데이터를 모두 저장할 수 있는 조건에서 시간 단계별 계산 예산을 제한합니다. uniform sampling이 검토한 복잡한 구성 요소보다 유리한 조건이 보고되지만, 이를 모든 데이터·예산·CL 설정의 보편적 순위로 확장하지 않습니다. ImageNet2K/CGLM, ImageNet-1K 사전학습 ResNet50, 20개 시간 단계가 원논문의 주요 실험 조건입니다. 현재 공통 CIFAR-100-224 첫 태스크 pilot은 이 논문의 재현 실험이 아닙니다. [논문 본문](https://arxiv.org/html/2303.11165v2)

sampling 비교의 기본 Naive는 uniform+CE이고, distillation·FC correction 비교에서는 class-balanced+CE를 Naive로 사용합니다. 서로 다른 표의 결과를 모두 같은 uniform 기준선의 결과로 해석하지 않아야 합니다. [논문의 비교 기준선](https://arxiv.org/html/2303.11165v2#S4)

논문은 forward/backward 비용에 따라 허용 **학습 iteration 수**를 조정합니다. 추가 teacher forward가 있는 distillation, 후보를 평가하는 sampling, FC-only 학습은 같은 예산에서 서로 다른 update 수를 얻습니다. 후보 forward 이후 샘플을 고르는 selector 자체의 비용은 negligible로 가정합니다. 이는 현재 도구의 wall time 측정이나 지원 operator의 부분 FLOPs 집계와 다른 비용 모델입니다. [계산 예산 정의와 실험 조건](https://arxiv.org/html/2303.11165v2#S2)

Mammoth에 추가할 때는 다음 범위를 권장합니다.

1. **기준선**: 해당 시간 단계까지 공개된 데이터 전체에서 균등하게 minibatch를 뽑아 CE로 학습하는 `budgetcl_uniform` 등의 엔트리. 기존 ER의 현재 minibatch+500개 replay 조건과 구별합니다. class-balanced CE 변형은 별도 설정으로 표시합니다.
2. **데이터 이력**: 모든 관측 샘플을 index/path로 추적하고 다음 단계의 샘플은 허용하지 않습니다. 예산 때문에 학습에 쓰지 않은 공개 샘플도 보관합니다. 기존 unlimited buffer 기능을 참고할 수 있지만, chunk 도착과 전체 이력 sampling을 연결해야 합니다.
3. **예산 제어**: 단계별 update 한도와 구성 요소별 forward/backward 비용을 명시합니다. `--fitting_mode iters --n_iters`는 loop 호출 수를 제한하는 도구로 참고할 수 있으며, 여러 내부 update나 teacher/SVD 등의 비용까지 자동으로 같은 예산으로 맞춰주지는 않습니다.
4. **후속 변형**: sampling·distillation·FC correction·FC-only를 같은 데이터 공개 조건에서 추가합니다. 기준선과 비용 모델을 먼저 검증한 뒤 변형을 늘리는 편이 결과를 해석하기 쉽습니다.

위 방향은 [저자의 데이터 이력 처리](https://github.com/drimpossible/BudgetCL/blob/7b5057feeb25fb58b3d507f2a878c134a698d678/src/datasets.py), [학습 구성과 동결 경로](https://github.com/drimpossible/BudgetCL/blob/7b5057feeb25fb58b3d507f2a878c134a698d678/src/main.py)를 Mammoth의 buffer·task hook·training loop와 비교한 제안입니다.

공개 코드 snapshot `7b5057f`에는 **정적 검토로 확인한 불일치**가 있습니다. `--distill`의 소문자 선택값과 loss 분기의 대문자 문자열이 다르고, ImageNet2K loader는 선택한 `curr_paths/curr_labels` 대신 저장된 전체 목록을 사용합니다. 따라서 공개 파일을 그대로 복사하기보다 의도한 선택·손실 경로를 native 구현과 의미 있는 테스트로 확인해야 합니다. 런타임 동작을 재현하여 검증한 결과는 아닙니다. [CLI 선택값](https://github.com/drimpossible/BudgetCL/blob/7b5057feeb25fb58b3d507f2a878c134a698d678/src/opts.py#L103), [loss 분기](https://github.com/drimpossible/BudgetCL/blob/7b5057feeb25fb58b3d507f2a878c134a698d678/src/training.py#L71), [ImageNet2K sampling 경로](https://github.com/drimpossible/BudgetCL/blob/7b5057feeb25fb58b3d507f2a878c134a698d678/src/datasets.py#L291)

`total_steps`도 teacher/sampler 비용에 맞춰 자동 조정되지 않습니다. 종료 조건이 optimizer step 뒤의 `i > total_steps`이므로 긴 loader에서는 설정값 C에 대해 C+2 updates를 수행할 수 있습니다. 이식 시에는 설정된 loop 한도와 실제 optimizer step 수를 대조해야 합니다. [종료 조건](https://github.com/drimpossible/BudgetCL/blob/7b5057feeb25fb58b3d507f2a878c134a698d678/src/training.py#L104)

## aL-SAR: ResNet 경로를 우선 이식

**aL**은 임의의 층을 각각 끄는 방식이 아니라 앞쪽의 연속 layer prefix를 동결합니다. block별 Fisher 정보와 연산 비용을 이용하고, 현재 batch의 gradient 정보와 backward 절감량을 함께 반영해 동결 범위를 선택합니다. **SAR**은 사용 횟수가 많은 샘플의 재선택을 줄이되, class gradient 유사도를 반영한 유효 사용 횟수로 선택 확률을 보정합니다. 고정 메모리는 class 균형을 유지하도록 교체하고 학습 minibatch는 메모리에서 가져옵니다. [aL과 SAR의 정의](https://arxiv.org/html/2410.15143v2#S3)

이식 대상으로는 저자 CIL 경로의 `main_new.py`와 `methods/ours_new.py`의 `Ours` 클래스를 검토했습니다. 코드와 논문의 기본값도 구별해야 합니다. 논문은 4 updates마다 전체 gradient를 갱신하고 EMA 0.01을 사용하지만, CIL 코드는 확률적 unfreeze와 EMA 0.001을 사용합니다. SAR용 좌표도 코드에서는 일부 conv 가중치의 0.01%, 논문에서는 전체 모델의 0.05%로 다릅니다. 이는 SAR용 선택이며, 코드의 Fisher는 unfrozen block 전체 파라미터의 minibatch `p.grad` 제곱 합으로 추정합니다. **논문 설정과 코드 snapshot 중 어느 쪽을 따르는지 명시해야 합니다.** [방법 선택 코드](https://github.com/snumprlab/budgeted-cl/blob/fb4d79cd928d5f00b013d3a2c525a5a1e374b337/budgeted_CL_CIL/utils/method_manager_new.py), [CIL 구현](https://github.com/snumprlab/budgeted-cl/blob/fb4d79cd928d5f00b013d3a2c525a5a1e374b337/budgeted_CL_CIL/methods/ours_new.py), [논문 부록](https://arxiv.org/html/2410.15143v2#A1)

실제 효율에 영향을 주는 이식 쟁점은 다음과 같습니다.

- **Backward 절감**: 검토한 일반 학습 경로는 forward 뒤 파라미터의 `requires_grad`를 변경하고 `loss.backward()`를 호출합니다. 활성 경로에 prefix 출력 detach나 `no_grad` 구간이 보이지 않으므로, 동결 block의 backward 비용을 장부에서 빼는 것과 실제 실행 절감을 동일하게 볼 수 없습니다. 계산 그래프·실행 kernel을 확인한 뒤 동결을 적용하는 시점과 경계를 결정해야 합니다. [학습·동결 경로](https://github.com/snumprlab/budgeted-cl/blob/fb4d79cd928d5f00b013d3a2c525a5a1e374b337/budgeted_CL_CIL/methods/ours_new.py#L393), [PyTorch의 requires_grad 동작](https://docs.pytorch.org/docs/2.14/notes/autograd.html#setting-requires-grad)
- **통계 계산 비용**: 저자의 autograd hook은 Linear/Conv2d의 per-example gradient를 생성한 뒤 선택 좌표를 사용합니다. 좌표가 작다는 사실이 gradient 생성도 그만큼 저렴하다는 의미는 아닙니다. peak memory와 통계 갱신 시간을 별도 phase로 기록해야 합니다. [per-example gradient 구현](https://github.com/snumprlab/budgeted-cl/blob/fb4d79cd928d5f00b013d3a2c525a5a1e374b337/budgeted_CL_CIL/utils/autograd_hacks.py#L127)
- **ViT/LoRA 범위**: 검토한 CIL 코드는 ViT에서 similarity retrieval을 생략하고 gradient 대상 선택은 CNN/ResNet에 맞춰져 있습니다. 논문의 ViT aL 실험만으로 완전한 ViT aL-SAR 이식이 준비됐다고 볼 수 없습니다. 전체 파라미터를 다시 켜는 unfreeze도 그대로 옮기면 PEFT의 frozen backbone을 풀 수 있으므로 원래 trainability mask를 보존해야 합니다. [gradient 대상·retrieval·unfreeze 구현](https://github.com/snumprlab/budgeted-cl/blob/fb4d79cd928d5f00b013d3a2c525a5a1e374b337/budgeted_CL_CIL/methods/ours_new.py#L151)
- **BatchNorm 상태**: 저자 경로는 `model.train()`에서 파라미터 flag만 변경하므로 동결 prefix의 running statistics도 계속 갱신됩니다. 이식 시 BatchNorm 정책과 source 설정과의 차이를 기록해야 합니다. [학습 모드](https://github.com/snumprlab/budgeted-cl/blob/fb4d79cd928d5f00b013d3a2c525a5a1e374b337/budgeted_CL_CIL/methods/ours_new.py#L393), [BatchNorm 동작](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html)

권장 파일 구성은 `models/alsar.py`, `models/alsar_utils/`, `models/config/alsar.yaml`입니다. replay count·similarity·Fisher·prefix 선택 상태를 checkpoint에 포함하고, `observe`의 loss 반환 계약과 `meta_begin_task/meta_observe/meta_end_task`를 유지합니다. 온라인 sample 도착, update credit, 실제 optimizer step 수는 별도로 기록하고 optimizer 재생성과 scheduler 갱신이 상태를 깨지 않도록 연결해야 합니다. 이 파일들은 아직 생성하지 않았습니다.

## 공통 데이터와 효율성 비교에 연결하는 방식

현재 실행 준비 조건인 **seq-cifar100-224, 첫 태스크 10 classes, clean labels, outer 1 epoch, batch 64, seed 1993**을 모든 후보의 자원 pilot에 사용할 수 있습니다. aL-SAR의 첫 이식은 ResNet 경로로 하고, 논문의 CIFAR ResNet32 등 원래 모델/입력과 다른 조건이라는 점을 기록하는 것을 권장합니다. BudgetCL의 전체 이력 보관과 aL-SAR의 고정 메모리를 섞어 단순 순위를 매기기보다, 원래 protocol 결과와 replay 조건을 통제한 비교를 구분해야 합니다.

현재 계측 도구는 시간·메모리·GPU 에너지 및 부분 FLOPs를 수집하도록 준비돼 있습니다. **BudgetCL/aL-SAR의 예산 controller는 구현하지 않았습니다.** 계산 예산 비교를 추가할 때는 다음을 별도로 정해야 합니다.

| 추가할 정의/기록 | 필요한 이유 |
|---|---|
| 공개된 sample 범위와 chunk/stream 도착 순서 | 학습에 쓰지 않은 공개 sample과 아직 미래인 sample을 구분 |
| 관측 sample·학습/replay sample·실제 optimizer step 수 | `meta_observe` 호출 수만으로 내부 update나 replay 비용을 셀 수 없음 |
| forward/backward·teacher·Fisher/SAR·통계·task-end 비용의 예산 장부 | 부분 profiler FLOPs를 완전한 계산 예산으로 쓰지 않도록 범위 명시 |
| replay·모델·optimizer·feature/logit·통계의 중복 없는 저장 bytes | aL-SAR의 저장 예산은 단순 buffer sample 수보다 넓음. 현재 snapshot의 겹치는 범주를 합산하면 안 됨 |
| 같은 backbone·사전학습·precision·데이터 공개/메모리 정책 | 동일 데이터셋만으로 방법 간 예산 공정성이 성립하지 않음 |
| 예산 소진 시 update 중단/허용 기준과 평가 비용의 처리 | 지출 범위와 비교 규칙을 실행 전 고정 |

aL-SAR 논문은 FLOPs와 전체 저장량을 함께 통제하며, 저자 counter는 operator별 forward/backward 추정과 수동 추가 비용을 사용합니다. 구현 후에는 이 추정값을 실제 kernel/시간 계측과 대조하고 GPU 에너지를 별도 결과로 기록해야 합니다. 저자 비용 추정치가 그대로 실제 벽시간이나 전력을 뜻하지 않습니다. [논문의 비용 정의](https://arxiv.org/html/2410.15143v2), [저자 operator counter](https://github.com/snumprlab/budgeted-cl/blob/fb4d79cd928d5f00b013d3a2c525a5a1e374b337/budgeted_CL_CIL/flops_counter/ptflops/pytorch_ops.py)

첫 태스크에서는 resource overhead와 warm-up 이후 동결·sampling 동작을 확인할 수 있지만, 장기 망각과 전체 stream의 평균 성능은 확인할 수 없습니다. 이후 후보가 확정되면 전체 sequence에서 최종 정확도·구간별 seen-class 정확도/평균·forgetting·저장량 증가와 계산 예산별 성능 곡선을 추가하는 방향을 권장합니다.

이 문서는 후보 검토이며 기존 70개 inventory·65개 실행 준비 계획을 수정하지 않습니다. W&B/Slack 연결은 기존 실행 도구가 담당하도록 유지하고, 방법 구현 및 protocol 확정 후 같은 결과 저장 경로에 추가하는 것이 후속 작업입니다.
