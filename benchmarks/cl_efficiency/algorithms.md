# 현재 CL 알고리즘과 특성

2026-10-08 `inventory()`가 `models/*.py`의 NAME·COMPATIBILITY를 AST로 확인한 결과는 **77개 소스 엔트리**입니다. 정적 추론 3개·joint 기준선 2개를 제외하면 학습 방법과 변형 72개입니다. 10월 1일의 70개에 EASE를 추가했던 71개 `inventory.csv`·`manifest.json` 스냅샷은 그대로 보존하며, 이후 HiDe-Prompt와 네 LoRA 방법, RainbowPrompt가 추가됐습니다. 이 수는 전체 모델의 import 성공이나 실험 재현을 뜻하지 않습니다. `models/lora_prototype_utils/`와 `models/low_rank_utils/`는 내부 구성요소이며 독립 모델 엔트리가 아닙니다.

표의 장단점과 비용 영향은 코드 구조에 근거한 분석입니다. EASE·RanPAC·L2P·DualPrompt의 [공통 첫 태스크 pilot](baselines/efficiency_pilot_report.md), [EASE 크기별 검증·측정](ease_sizes/report.md), [HiDe-Prompt Base 전체10태스크](hide_prompt/full_cl/report.md)와 [Small/Tiny 단기 결과](hide_sizes/report/report.md)는 각 보고서의 조건으로 해석합니다. 전체77개 엔트리의 공통 정확도·효율성 비교는 완료하지 않았습니다. 기존 실행 인자·외부 요구사항은 71개 `manifest.json` 스냅샷에, 추가 방법의 조건은 각 구현·검증 문서에 기록합니다.

RainbowPrompt는 공식 활성 경로를 구현하고 CPU 검증을 완료했습니다. learned gate는 포함하지 않으며 CIFAR 정확도·GPU 효율성은 아직 검증하지 않았습니다. [구현 범위와 계획](rainbow_prompt/README.md), [CPU 검사](rainbow_prompt/CPU_VALIDATION.md)를 참고합니다.

EWC-LoRA·SD-LoRA·InfLoRA·CL-LoRA는 Base ViT 첫2태스크의 저자·native 총8회 검증에서 네 정확도 지표·14개 구조 검사·첫 배치 입력 검사가 모두 pass입니다. EWC·SD·CL의 네 정답 수는 저자 실행과 같으며, Inf의 태스크2 저자−native 차이는 전체 −0.15%p·과거 −0.40%p·신규 +0.10%p입니다. [검증 결과](low_rank/RESULTS.md)와 [구현·실행 범위](low_rank/README.md)를 참고합니다. 공통 저자 저장소 변형을 검증했으며 논문 수치 재현·효율성 순위·전체10태스크 성능은 평가하지 않았습니다. LoRA Small·Tiny는 CPU 생성·forward 확인 범위입니다.

## 알고리즘별 비교

| 모델 | 계열 | 특징 | 장점 | 단점과 효율성 영향 |
|---|---|---|---|---|
| [agem](../../models/agem.py) | 그래디언트 제약 | 과거 reference gradient에 대한 평균 제약 투영 | GEM보다 저렴한 projection | reference gradient 계산과 buffer 필요 |
| [agem-r](../../models/agem_r.py) | 그래디언트 제약 | A-GEM에 reservoir sampling buffer 적용 | 스트림에서 일관된 sampling | 추가 backward와 projection |
| [attriclip](../../models/attriclip.py) | Prompt와 CLIP | attribute bank에서 선택한 text prompt로 CLIP 적응 | 클래스 속성 재사용과 적은 학습 파라미터 | 저장소 자체에 원논문 성능 미재현 경고가 있음 |
| [bic](../../models/bic.py) | 리플레이와 증류 | 증류 후 별도 validation 샘플로 bias correction | 새 클래스 bias를 직접 보정 | 검증용 buffer 분할과 추가 보정 학습 |
| [ccic](../../models/ccic.py) | 준지도 CL | contrastive/interpolation consistency와 replay | 부분 라벨 스트림을 지원 | augmentation/contrastive 연산; 이번 fully labeled pilot은 CSSL 효능 미평가 |
| [cgil](../../models/cgil.py) | Prompt와 CLIP | 클래스 feature 분포 생성 모델과 prompt 정렬 | 원본 이미지를 저장하지 않고 생성 feature replay | 생성 모델 및 태스크 말 prompt 정렬 비용 |
| [clip](../../models/clip.py) | 정적 추론 기준선 | 학습 없이 CLIP image/text similarity | 학습 비용 0; 사전학습 기준점 | backbone 메모리와 추론 비용은 남으며 CL 적응은 없음 |
| [cnll](../../models/cnll.py) | 노이즈 CL | delayed/noisy/clean buffer와 선별 및 재학습 | 노이즈 라벨의 학습을 제어 | 여러 buffer와 SSL/fitting 루프; true_labels extra field 필요 |
| [coda-prompt](../../models/coda_prompt.py) | Prompt | attention 가중합으로 prompt 구성요소 결합 | differentiable prompt 선택과 지식 결합 | pool 구성/직교 초기화와 attention 비용 |
| [dap](../../models/dap.py) | Prompt | instance별 prompt를 생성하는 모듈 | 입력별 적응과 원본 replay 불필요 | prompt generator 연산; 원논문 split/checkpoint와 재현 조건 차이 |
| [der](../../models/der.py) | 리플레이 | 저장된 과거 logits를 distillation | 출력 함수 보존과 단순한 구현 | 샘플 외에 logits도 저장 |
| [derpp](../../models/derpp.py) | 리플레이 | DER logits distillation과 replay label CE | 안정적이고 강력한 비교 기준선 | 추가 replay forward와 두 손실 가중치 |
| [derpp-casper](../../models/derpp_casper.py) | 리플레이 | DER logits distillation과 replay label CE; CaSpeR: buffer feature graph의 spectral 정규화 | 안정적이고 강력한 비교 기준선; 클래스 feature 구조 보존 | 추가 replay forward와 두 손실 가중치; kNN graph/eigenvalue 연산 추가 |
| [derpp-cscct](../../models/derpp_cscct.py) | 리플레이 | DER logits distillation과 replay label CE; CSCCT: cross-space clustering와 cross-task consistency | 안정적이고 강력한 비교 기준선; 표현 구조와 과거 태스크 일관성 | 추가 replay forward와 두 손실 가중치; feature/teacher 연산과 손실 추가 |
| [derpp-lider](../../models/derpp_lider.py) | 리플레이 | DER logits distillation과 replay label CE; LiDER: replay buffer에서 Lipschitz regularization | 안정적이고 강력한 비교 기준선; 작은 buffer의 과적합을 제어 | 추가 replay forward와 두 손실 가중치; 중간 feature 및 power iteration 비용; 가중치가 0이면 효과 없음 |
| [derpp-star](../../models/derpp_star.py) | 리플레이 | DER logits distillation과 replay label CE; STAR: 가중치 perturbation을 이용한 안정성 정규화 | 안정적이고 강력한 비교 기준선; 과거 지식의 안정성을 보강 | 추가 replay forward와 두 손실 가중치; 추가 perturbation/gradient 연산; STARPrompt와 다른 방법 |
| [dualprompt](../../models/dualprompt.py) | Prompt | 공유 G-prompt와 태스크 특화 E-prompt | 공유/특화 지식 분리 | prompt 선택 및 frozen feature 추출 비용 |
| [ease](../../models/ease.py) | 확장형 Adapter | frozen ViT에 태스크별 adapter를 추가하고 raw prototype과 semantic 보완으로 분류 | 과거 원본 이미지 replay 없이 완료한 adapter를 보존 | adapter·prototype 저장량 증가, 태스크 말과 추론에서 adapter별 transformer 반복 |
| [er](../../models/er.py) | 리플레이 | 과거 원본 샘플을 buffer에서 재학습 | 강력하고 이해하기 쉬운 기준선 | 원본 데이터 저장과 추가 배치 연산 |
| [er-ace](../../models/er_ace.py) | 리플레이 | 현재 배치의 unseen/old class 간섭을 비대칭 CE로 제어 | 클래스 간섭을 줄이는 간단한 확장 | buffer가 필요하고 첫 태스크에는 효과가 충분히 드러나지 않음 |
| [er-ace-aer-abs](../../models/er_ace_aer_abs.py) | 노이즈 CL | alternate replay와 asymmetrical buffer sampling | 라벨 노이즈의 악영향을 완화 | buffer fitting 비용과 태스크별 스케줄 영향 |
| [er-ace-casper](../../models/er_ace_casper.py) | 리플레이 | 현재 배치의 unseen/old class 간섭을 비대칭 CE로 제어; CaSpeR: buffer feature graph의 spectral 정규화 | 클래스 간섭을 줄이는 간단한 확장; 클래스 feature 구조 보존 | buffer가 필요하고 첫 태스크에는 효과가 충분히 드러나지 않음; kNN graph/eigenvalue 연산 추가 |
| [er-ace-cscct](../../models/er_ace_cscct.py) | 리플레이 | 현재 배치의 unseen/old class 간섭을 비대칭 CE로 제어; CSCCT: cross-space clustering와 cross-task consistency | 클래스 간섭을 줄이는 간단한 확장; 표현 구조와 과거 태스크 일관성 | buffer가 필요하고 첫 태스크에는 효과가 충분히 드러나지 않음; feature/teacher 연산과 손실 추가 |
| [er-ace-lider](../../models/er_ace_lider.py) | 리플레이 | 현재 배치의 unseen/old class 간섭을 비대칭 CE로 제어; LiDER: replay buffer에서 Lipschitz regularization | 클래스 간섭을 줄이는 간단한 확장; 작은 buffer의 과적합을 제어 | buffer가 필요하고 첫 태스크에는 효과가 충분히 드러나지 않음; 중간 feature 및 power iteration 비용; 가중치가 0이면 효과 없음 |
| [er-ace-star](../../models/er_ace_star.py) | 리플레이 | 현재 배치의 unseen/old class 간섭을 비대칭 CE로 제어; STAR: 가중치 perturbation을 이용한 안정성 정규화 | 클래스 간섭을 줄이는 간단한 확장; 과거 지식의 안정성을 보강 | buffer가 필요하고 첫 태스크에는 효과가 충분히 드러나지 않음; 추가 perturbation/gradient 연산; STARPrompt와 다른 방법 |
| [er-ace-tricks](../../models/er_ace_tricks.py) | 리플레이 | ER-ACE에 bias correction/decay/sampling tricks 추가 | ER-ACE 구성요소 비교 | 추가 튜닝 및 task-end correction 비용 |
| [er-star](../../models/er_star.py) | 리플레이 | 과거 원본 샘플을 buffer에서 재학습; STAR: 가중치 perturbation을 이용한 안정성 정규화 | 강력하고 이해하기 쉬운 기준선; 과거 지식의 안정성을 보강 | 원본 데이터 저장과 추가 배치 연산; 추가 perturbation/gradient 연산; STARPrompt와 다른 방법 |
| [er-tricks](../../models/er_tricks.py) | 리플레이 | ER에 bias correction/learning-rate decay/sampling tricks 추가 | 작은 구성 변경으로 baseline 개선 | 여러 tricks의 상호작용과 추가 보정 학습 |
| [ewc-on](../../models/ewc_on.py) | 가중치 정규화 | 누적 Fisher로 중요한 파라미터의 변화 억제 | 원본 replay 샘플을 장기 저장하지 않음 | Fisher/이전 가중치 저장과 Fisher 계산 |
| [fdr](../../models/fdr.py) | 리플레이 | 저장된 예제의 출력 함수 거리 정규화 | 함수 변화를 직접 제어 | 샘플 및 출력 저장과 추가 forward |
| [first-stage-starprompt](../../models/first_stage_starprompt.py) | Prompt와 CLIP | STARPrompt의 CLIP semantic key 단계만 사용 | 첫 단계 비용과 기여를 분리 | 전체 STARPrompt의 시각 prompt 적응을 대체하지 못함 |
| [gdumb](../../models/gdumb.py) | buffer 기준선 | buffer를 채운 후 매 태스크 끝에서 모델을 새로 학습 | 샘플 선택의 효과를 분리 | end_task의 fitting_epochs=256 기본값이 매우 비쌈 |
| [gdumb-lider](../../models/gdumb_lider.py) | buffer 기준선 | buffer를 채운 후 매 태스크 끝에서 모델을 새로 학습; LiDER: replay buffer에서 Lipschitz regularization | 샘플 선택의 효과를 분리; 작은 buffer의 과적합을 제어 | end_task의 fitting_epochs=256 기본값이 매우 비쌈; 중간 feature 및 power iteration 비용; 가중치가 0이면 효과 없음 |
| [gem](../../models/gem.py) | 그래디언트 제약 | 태스크별 과거 gradient 제약을 QP로 해결 | 과거 손실 제약을 명시적으로 적용 | 태스크별 backward와 QP solver 비용 |
| [gss](../../models/gss.py) | 샘플 선택 | gradient 다양성으로 buffer 샘플 선별 | 작은 buffer의 대표성 개선 | 샘플 선택 자체의 여러 gradient 계산 |
| [hal](../../models/hal.py) | 샘플 선택 | 태스크별 anchor 생성과 replay | 망각 제약에 anchor 활용 | 태스크 말 anchor 최적화와 추가 저장 |
| [icarl](../../models/icarl.py) | 리플레이와 증류 | exemplar herding과 클래스 평균 분류 및 distillation | 대표 exemplar와 incremental classifier | 태스크 말 herding과 feature 수집 비용 |
| [icarl-casper](../../models/icarl_casper.py) | 리플레이와 증류 | exemplar herding과 클래스 평균 분류 및 distillation; CaSpeR: buffer feature graph의 spectral 정규화 | 대표 exemplar와 incremental classifier; 클래스 feature 구조 보존 | 태스크 말 herding과 feature 수집 비용; kNN graph/eigenvalue 연산 추가 |
| [icarl-cscct](../../models/icarl_cscct.py) | 리플레이와 증류 | exemplar herding과 클래스 평균 분류 및 distillation; CSCCT: cross-space clustering와 cross-task consistency | 대표 exemplar와 incremental classifier; 표현 구조와 과거 태스크 일관성 | 태스크 말 herding과 feature 수집 비용; feature/teacher 연산과 손실 추가 |
| [icarl-lider](../../models/icarl_lider.py) | 리플레이와 증류 | exemplar herding과 클래스 평균 분류 및 distillation; LiDER: replay buffer에서 Lipschitz regularization | 대표 exemplar와 incremental classifier; 작은 buffer의 과적합을 제어 | 태스크 말 herding과 feature 수집 비용; 중간 feature 및 power iteration 비용; 가중치가 0이면 효과 없음 |
| [idefics](../../models/idefics.py) | 정적 추론 기준선 | IDEFICS로 이미지 클래스 이름 생성 | multimodal zero-shot 비교 | 큰 모델과 생성 latency; 학습 CL 방법이 아님 |
| [joint](../../models/joint.py) | 오프라인 기준선 | 모든 태스크 데이터로 joint training | 정확도 상한 비교에 유용 | 과거 데이터 접근이 필요하고 첫 태스크 제한과 충돌 |
| [joint-gcl](../../models/joint_gcl.py) | 오프라인 기준선 | GCL 스트림 데이터를 모아 joint training | GCL 오프라인 상한 기준선 | 표준 1개 태스크 개념과 다르며 스트림 전체를 모음 |
| [l2p](../../models/l2p.py) | Prompt | frozen ViT에서 query로 prompt pool top-k 선택 | 작은 학습 파라미터와 원본 replay 불필요 | query용 backbone과 prompt token 연산; selection 오차 |
| [llava](../../models/llava.py) | 정적 추론 기준선 | LLaVA로 이미지 클래스 이름 생성 | 자연어 지식을 활용한 zero-shot 비교 | 생성 추론 지연과 큰 모델; 학습 CL 방법이 아님 |
| [lucir](../../models/lucir.py) | 리플레이와 증류 | cosine classifier와 feature/순위 distillation | old/new class의 균형 개선 | 여러 손실과 태스크 말 fine-tuning |
| [lwf](../../models/lwf.py) | 증류 | 이전 모델의 현재 입력에 대한 예측 증류 | 과거 원본 샘플 없이 동작 | teacher 모델과 추가 forward; 큰 domain shift에 취약 |
| [lwf-mc](../../models/lwf_mc.py) | 증류 | iCaRL 방식의 multi-class LwF 변형 | incremental class sigmoid distillation | teacher 보관과 calibration 부담 |
| [lws](../../models/lws.py) | 편향 CL | shortcut와 cluster 지식의 replay 및 weighting | spurious correlation이 있는 biased-class-il에 맞춤 | CelebA 준비와 cluster overhead; 일반 CIFAR와 직접 비교 불가 |
| [mer](../../models/mer.py) | 메타 학습 | replay batch의 Reptile식 meta update | 태스크 간 gradient 정렬 유도 | inner update 반복으로 처리량 저하 가능 |
| [moe-adapters](../../models/moe_adapters.py) | Adapter와 CLIP | CLIP에 전문가 adapter와 routing 모듈 추가 | backbone 고정과 전문가 분업 | 전문가/routing 연산과 task progression 저장 비용 |
| [pnn](../../models/pnn.py) | 모델 확장 | 태스크마다 column을 추가하고 이전 column 고정 | 과거 column의 가중치 망각 방지 | 태스크별 모델 증가; 이 구현은 task-il 전용 |
| [puridiver](../../models/puridiver.py) | 노이즈 CL | 손실과 feature 다양성으로 replay 샘플 정제 | 노이즈/blurred stream에 맞는 선택 | 매 batch feature 선택과 buffer fitting; true_labels 필요 |
| [ranpac](../../models/ranpac.py) | 사전학습과 해석적 분류기 | 첫 태스크 적응 후 frozen feature의 random projection와 ridge | 후속 태스크의 backbone backward 제거 | 첫 태스크는 실제 학습; rp_size=10000의 Gram/solve가 무거움 |
| [rainbow-prompt](../../models/rainbow_prompt.py) | Prompt 변환·정렬 | 현재/이전 base prompt를 변환하여 결합하고 저장 prompt를 batch 평균 key로 선택 | 원본 replay 없이 prompt 간 지식 결합; 전체 gate 복원과 분리된 공식 활성 경로 대조 가능 | query·prompt 두 ViT forward와 반복 attention/FFN, 태스크별 cache 증가, batch 구성에 따른 routing 변화. 공식 활성 코드의 learned gate 미포함; CPU 검증만 완료 |
| [rpc](../../models/rpc.py) | 리플레이 | Regular Polytope Classifier로 클래스 표현 구조 고정 | 클래스 간 기하 구조를 제어 | buffer 필요 및 기하 가정의 데이터 의존성 |
| [second-order](../../models/second_order.py) | LoRA와 정규화 | LoRA/IA3/full tuning 및 ITA/IEL second-order 제약 | 기존 LoRA/IA3 기반과 model composition 비교 가능 | pre-tuning, Fisher, 분포 및 정렬 비용을 포함해야 함 |
| [second-stage-starprompt](../../models/second_stage_starprompt.py) | Prompt와 CLIP | STARPrompt의 ViT residual prompt 단계만 사용 | 시각 prompt 단계의 ablation | 키 checkpoint 또는 CLIP template에 성능 의존 |
| [sgd](../../models/sgd.py) | 기준선 | 새 데이터에 순차 fine-tuning | 가장 단순하고 추가 저장공간이 작음 | 과거 태스크 망각이 큼 |
| [si](../../models/si.py) | 가중치 정규화 | 학습 경로 기반 파라미터 중요도 누적 | 별도 Fisher 데이터 패스 없이 중요도 계산 | 모든 학습 파라미터의 중요도와 경로 통계 저장 |
| [slca](../../models/slca.py) | 사전학습과 분류기 정렬 | backbone을 낮은 LR로 학습 후 클래스 Gaussian으로 classifier alignment | 표현과 classifier bias를 따로 제어 | 기본은 full backbone 학습으로 PEFT가 아님; 분포/정렬 비용 |
| [spr](../../models/spr.py) | 노이즈 CL | self-supervised expert로 buffer를 정제 후 재학습 | 노이즈 스트림에서 replay 품질 개선 | expert/base 모델과 3000/4000회 inner 학습 기본값 비용 |
| [starprompt](../../models/starprompt.py) | Prompt와 CLIP | CLIP semantic key 학습과 ViT residual prompt의 2단계 | semantic routing과 시각 적응 결합 | CLIP+ViT 상주와 begin/end_task 내부 학습 |
| [tak](../../models/tak.py) | 모델 병합과 정규화 | CLIP task vector를 KFAC/Fisher 제약으로 학습 및 병합 | 병합 후 단일 backbone 추론 가능; 선택적 LoRA | 기본은 광범위 fine-tuning; Fisher/KFAC와 task vector 저장 비용 |
| [twf](../../models/twf.py) | 리플레이와 증류 | 사전학습 teacher의 feature를 attention distillation | 중간 표현과 사전학습 지식 보존 | teacher와 attention mask 저장; checkpoint 조건 영향 |
| [xder](../../models/xder.py) | 리플레이 | 과거 logits 수정 및 미래/현재 클래스 일관성 제약 | 클래스 간 균형과 전이 개선 | DER++보다 복잡한 손실과 buffer 갱신 비용 |
| [xder-ce](../../models/xder_ce.py) | 리플레이 | X-DER의 CE 기반 변형 | X-DER 구성요소 비교에 유용 | X-DER와 별개의 신규 알고리즘으로 세면 중복 |
| [xder-rpc](../../models/xder_rpc.py) | 리플레이 | X-DER와 Regular Polytope Classifier | 고정된 클래스 기하를 활용 | 분류기 가정과 replay overhead |
| [xder-rpc-casper](../../models/xder_rpc_casper.py) | 리플레이 | X-DER와 Regular Polytope Classifier; CaSpeR: buffer feature graph의 spectral 정규화 | 고정된 클래스 기하를 활용; 클래스 feature 구조 보존 | 분류기 가정과 replay overhead; kNN graph/eigenvalue 연산 추가 |
| [xder-rpc-cscct](../../models/xder_rpc_cscct.py) | 리플레이 | X-DER와 Regular Polytope Classifier; CSCCT: cross-space clustering와 cross-task consistency | 고정된 클래스 기하를 활용; 표현 구조와 과거 태스크 일관성 | 분류기 가정과 replay overhead; feature/teacher 연산과 손실 추가 |
| [xder-rpc-star](../../models/xder_rpc_star.py) | 리플레이 | X-DER와 Regular Polytope Classifier; STAR: 가중치 perturbation을 이용한 안정성 정규화 | 고정된 클래스 기하를 활용; 과거 지식의 안정성을 보강 | 분류기 가정과 replay overhead; 추가 perturbation/gradient 연산; STARPrompt와 다른 방법 |
| [zscl](../../models/zscl.py) | CLIP 증류 | reference image/text 증류와 weight averaging으로 zero-shot 보존 | 학습 후 zero-shot 능력 유지 목적 | CLIP teacher와 Conceptual Captions 외부 데이터 필요; 기본은 full tuning |
| [hide-prompt](../../models/hide_prompt.py) | 태스크 추론과 Prompt | TII·prompt 학습을 분리하고 feature 통계로 분류기를 정렬 | 원본 이미지 replay 없이 추론한 태스크에 맞는 prompt 선택 | 두 ViT 경로·통계 수집·분류기 정렬·routing 비용; 태스크 추론 오차의 영향 |
| [ewc-lora](../../models/ewc_lora.py) | LoRA와 Fisher 정규화 | frozen ViT의 새 LoRA를 누적 Fisher로 제약하고 태스크 평가 후 인자를 누적 | 과거 원본 이미지 buffer 없이 저차원 적응과 중요도 보존 | 층별 Fisher 행렬 저장·태스크 말 추가 데이터 패스·누적/reset 비용 |
| [sd-lora](../../models/sd_lora.py) | LoRA와 스케일 조절 | 태스크별 LoRA와 분류기를 추가하고 과거 LoRA의 기여도를 scaling factor로 조절 | 이전 LoRA 인자를 고정하며 현재 태스크에 필요한 기여도를 학습 | 태스크별 인자 저장과 과거 LoRA 계산 증가; scaling 변화가 과거 출력에 영향 |
| [inf-lora](../../models/inf_lora.py) | LoRA와 부분공간 | SVD로 A를 초기화·고정하고 DualGPM으로 과거 feature 부분공간을 보존하며 B를 학습 | 원본 이미지 replay 없이 저차원 업데이트의 간섭을 제어 | 태스크 시작·종료 통계 패스·CPU SVD·부분공간 저장; float32 투영의 수치 차이 |
| [cl-lora](../../models/cl_lora.py) | 공유·태스크별 LoRA와 증류 | 공유/특화 LoRA에 KD·gradient 재할당·직교 손실과 prototype 분류기를 적용 | 공유 지식과 태스크별 적응을 결합하며 원본 replay buffer 불필요 | 후속 태스크 KD·CE 별도 optimizer update·이전 공유 LoRA 저장·태스크별 prototype 추출 비용 |

## 지원 설정과 실행 그룹

아래의 첫 태스크 pilot은 준비한 조건이며 측정 완료를 뜻하지 않습니다. 기존 71개 manifest와 계획은 당시 스냅샷입니다. HiDe-Prompt와 네 LoRA 방법의 완료 범위는 별도 보고서에 따르며, LoRA 검증의 자원 관측을 효율성 pilot이나 방법 간 순위로 사용하지 않습니다.

| 모델 | 코드의 COMPATIBILITY | 실행 그룹 | 데이터셋 | 실행 조건 |
|---|---|---|---|---|
| `agem` | class-il, domain-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `agem-r` | class-il, domain-il, task-il, general-continual | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `attriclip` | class-il, domain-il, task-il, general-continual | pretrained_224 | seq-cifar100-224 | 첫 태스크 pilot |
| `bic` | class-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `ccic` | class-il, domain-il, task-il, cssl | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `cgil` | class-il, domain-il, task-il, general-continual | pretrained_224 | seq-cifar100-224 | 첫 태스크 pilot |
| `clip` | class-il, domain-il, task-il, general-continual | pretrained_224 | seq-cifar100-224 | 추론만 수행 |
| `cnll` | class-il, task-il | noisy_clean_224 | seq-cifar100-224 | Hard augmentation fixes crops at 32 pixels; adapt this code before comparing under the common 224-pixel protocol. |
| `coda-prompt` | class-il, task-il | pretrained_224 | seq-cifar100-224 | 첫 태스크 pilot |
| `dap` | class-il, task-il | pretrained_224 | seq-cifar100-224 | 첫 태스크 pilot |
| `der` | class-il, domain-il, task-il, general-continual | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `derpp` | class-il, domain-il, task-il, general-continual | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `derpp-casper` | class-il, domain-il, task-il, general-continual | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `derpp-cscct` | class-il, domain-il, task-il, general-continual | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `derpp-lider` | class-il, domain-il, task-il, general-continual | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `derpp-star` | class-il, domain-il, task-il, general-continual | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `dualprompt` | class-il, task-il | pretrained_224 | seq-cifar100-224 | 동일 NPZ·전처리 첫 태스크 측정3회·별도 부분 FLOPs1회 완료; 전체 CL 미검증 |
| `ease` | class-il, task-il | pretrained_224 | seq-cifar100-224 | Base 첫2태스크 검증·전체10태스크 완료; Small64/Tiny64/Small32/Tiny16 첫2태스크 검증·첫 태스크 pilot 완료 |
| `er` | class-il, domain-il, task-il, general-continual | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `er-ace` | class-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `er-ace-aer-abs` | class-il, task-il | noisy_clean_224 | seq-cifar100-224 | Shared clean-label pilot; native wrapper supplies true_labels=labels. No noisy-label robustness claim. |
| `er-ace-casper` | class-il, domain-il, task-il, general-continual | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `er-ace-cscct` | class-il, domain-il, task-il, general-continual | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `er-ace-lider` | class-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `er-ace-star` | class-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `er-ace-tricks` | class-il, domain-il, task-il, general-continual | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `er-star` | class-il, domain-il, task-il, general-continual | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `er-tricks` | class-il, domain-il, task-il, general-continual | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `ewc-on` | class-il, domain-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `fdr` | class-il, domain-il, task-il, general-continual | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `first-stage-starprompt` | class-il, domain-il, task-il, general-continual | pretrained_224 | seq-cifar100-224 | 첫 태스크 pilot |
| `gdumb` | class-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `gdumb-lider` | class-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `gem` | class-il, domain-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `gss` | class-il, domain-il, task-il, general-continual | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `hal` | class-il, domain-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `icarl` | class-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `icarl-casper` | class-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `icarl-cscct` | class-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `icarl-lider` | class-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `idefics` | class-il, domain-il, task-il, general-continual | pretrained_224 | seq-cifar100-224 | Large multimodal checkpoint, transformers/accelerate; llava defaults to 4-bit bitsandbytes |
| `joint` | class-il, domain-il, task-il | classic_resnet224 | seq-cifar100-224 | main.py forbids --stop_after for joint; all-task offline baseline needs a separate protocol. |
| `joint-gcl` | general-continual | classic_resnet224 | seq-cifar100-224 | General-continual only; joint GCL stream has no comparable single-task protocol. |
| `l2p` | class-il, domain-il, task-il, general-continual | pretrained_224 | seq-cifar100-224 | 동일 NPZ·전처리 첫 태스크 측정3회·별도 부분 FLOPs1회 완료; 전체 CL 미검증 |
| `llava` | class-il, domain-il, task-il, general-continual | pretrained_224 | seq-cifar100-224 | Large multimodal checkpoint, transformers/accelerate; llava defaults to 4-bit bitsandbytes |
| `lucir` | class-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `lwf` | class-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `lwf-mc` | class-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `lws` | biased-class-il | unsupported_biased | seq-cifar100-224 | Requires biased-class-il and bias/index fields; not compatible with the shared CIFAR100 class-il protocol. |
| `mer` | class-il, domain-il, task-il, general-continual | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `moe-adapters` | class-il, domain-il, task-il, general-continual | pretrained_224 | seq-cifar100-224 | 첫 태스크 pilot |
| `pnn` | task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `puridiver` | class-il, task-il | noisy_clean_224 | seq-cifar100-224 | Constructor only accepts seq-cifar10/seq-cifar100 and uses 32-pixel hard crops; seq-cifar100-224 requires an explicit port. |
| `ranpac` | class-il, domain-il, task-il | pretrained_224 | seq-cifar100-224 | 동일 NPZ·전처리 첫 태스크 측정3회·별도 부분 FLOPs1회 완료; 전체 CL 미검증 |
| `rpc` | class-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `second-order` | class-il, domain-il, task-il, general-continual | pretrained_224 | seq-cifar100-224 | 첫 태스크 pilot |
| `second-stage-starprompt` | class-il, domain-il, task-il, general-continual | pretrained_224 | seq-cifar100-224 | 첫 태스크 pilot |
| `sgd` | class-il, domain-il, task-il, general-continual | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `si` | class-il, domain-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `slca` | class-il, domain-il, task-il | pretrained_224 | seq-cifar100-224 | 첫 태스크 pilot |
| `spr` | class-il, task-il | noisy_clean_224 | seq-cifar100-224 | Shared clean-label pilot; native wrapper supplies true_labels=labels. No noisy-label robustness claim. |
| `starprompt` | class-il, domain-il, task-il, general-continual | pretrained_224 | seq-cifar100-224 | 첫 태스크 pilot |
| `tak` | class-il, domain-il, task-il, general-continual | pretrained_224 | seq-cifar100-224 | 첫 태스크 pilot |
| `twf` | class-il, task-il | classic_resnet224 | seq-cifar100-224 | Pilot uses the initial backbone as teacher; paper checkpoint is not supplied/downloaded by this plan |
| `xder` | class-il, domain-il, task-il, general-continual | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `xder-ce` | class-il, domain-il, task-il, general-continual | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `xder-rpc` | class-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `xder-rpc-casper` | class-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `xder-rpc-cscct` | class-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `xder-rpc-star` | class-il, task-il | classic_resnet224 | seq-cifar100-224 | 첫 태스크 pilot |
| `zscl` | class-il, domain-il, task-il, general-continual | pretrained_224 | seq-cifar100-224 | data/conceptual_captions must be prepared externally |
| `hide-prompt` | class-il, task-il | 개별 HiDe 검증·측정 | seq-cifar100-224 | Base 첫2태스크 검증·전체10태스크, Small/Tiny 첫2태스크 검증·첫 태스크 pilot 완료 |
| `ewc-lora` | class-il, task-il | 개별 low_rank 검증 | seq-cifar100-224 | Base 첫2태스크 검증 pass; 20epoch/task; Small/Tiny CPU 생성·forward만 확인 |
| `sd-lora` | class-il, task-il | 개별 low_rank 검증 | seq-cifar100-224 | Base 첫2태스크 검증 pass; 20epoch/task; Small/Tiny CPU 생성·forward만 확인 |
| `inf-lora` | class-il, task-il | 개별 low_rank 검증 | seq-cifar100-224 | Base 첫2태스크 검증 pass; 20epoch/task; Small/Tiny CPU 생성·forward만 확인 |
| `cl-lora` | class-il, task-il | 개별 low_rank 검증 | seq-cifar100-224 | Base 첫2태스크 검증 pass; 30epoch/task; Small/Tiny CPU 생성·forward만 확인 |
