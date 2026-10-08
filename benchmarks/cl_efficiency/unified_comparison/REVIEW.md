# 11개 방법 공통 비교: 최종 확인안

현재 상태: 최종 실행 확인 대기. 이 문서와 `review_plan.json`은 검토용 계획이며 GPU 실행을 허용하는 실행 파일이 아니다. 이 새 프로토콜의 GPU 실험은 아직 시작하지 않았다.

## 대상과 조건

| 계열 | 방법 |
|---|---|
| 일반 CL | Fine-tuning (`sgd`, optimizer는 AdamW), ER, DER++, LwF-MC |
| Prompt | CODA-Prompt, HiDe-Prompt |
| Adapter | EASE |
| Feature | RanPAC |
| LoRA | EWC-LoRA, InfLoRA, CL-LoRA |

각 방법에 Tiny16, Tiny64, Small16, Small64, Base16, Base64의 6개 조건을 적용한다. 총 66조건이며 `review_cells.csv`에 전체 목록을 저장했다.

- 데이터: CIFAR-100 / `seq-cifar100-224`, Class-IL. 기존 10태스크·태스크당 10클래스 구조를 유지하고 첫 2태스크만 실행한다. 클래스 순서 0–99, seed 1993.
- 백본: ViT-Ti/16, ViT-S/16, ViT-B/16. Tiny가 기본이며 Small/Base는 효율성 pilot이다. 동일 크기의 모든 방법은 같은 AugReg IN21K→IN1K 로컬 NPZ와 canonical encoder digest를 사용한다. 크기별 사전학습 recipe 차이는 기록한다.
- 학습 전처리: RandomResizedCrop224, scale 0.08–1, ratio 0.75–4/3, bicubic, antialias, horizontal flip 0.5, mean/std 0.5.
- 평가 전처리: Resize256 bicubic/antialias, CenterCrop224, mean/std 0.5. 평가 batch는 모든 조건에서 64. 과거·신규·전체 seen-class 정확도를 기록하며 정답 task ID를 제공하지 않는다.
- 실제 입력과 원본 샘플 ID, 증강·순서·보조 pass의 RNG를 점검한다. Replay 및 통계 수집 경로도 적용된 전처리를 기록한다.
- ER/DER++ buffer는 500장. 현재 batch16→replay16, 현재 batch64→replay64. DER++의 두 독립 replay draw와 추가 forward도 기록한다. 500장은 총 보조 메모리의 동일성을 뜻하지 않는다.
- EASE adapter bottleneck64, RanPAC 내부 bottleneck64 / RP dimension10000, LoRA rank10을 모든 크기에서 유지한다. HiDe prefix5·layer0–4·TII 폭 비율×2, CODA prompt length8·pool100. 비율 정렬 및 rank/prompt sweep은 후속이다.
- 일반 CL 4종은 새 ViT 기준선 recipe AdamW, LR1e-4, weight decay0. DER++ alpha0.3/beta0.5, LwF-MC wd_reg0. 다른 방법은 기존 방법별 optimizer/LR/scheduler와 batch scaling을 유지한다. 자세한 값은 `review_plan.json`에 저장했다.
- FP32, autocast off, matmul highest/TF32 false, cuDNN TF32 true, deterministic true/benchmark false. CPU thread2, loader workers4.
- GPU0·1에 독립 실험을 배치하되 GPU당 동시 실험은 1개. 같은 조건의 3회는 같은 물리 GPU에서 실행한다. UUID·CPU/I/O 경합을 기록하며 외부 GPU 작업이 있으면 해당 GPU의 순수 자원 계측을 시작하지 않는다.
- 공유 데이터 `/mnt/sdab1/share/data`, 공유 캐시 `/mnt/sdab1/userHome/wslim/.cache` 및 기존 torch/Hugging Face/CLIP 하위 경로를 재사용한다.

## 순서와 횟수

| 단계 | 조건당 실행 | 태스크와 명목 epoch | 계획 실행 수 |
|---|---|---|---:|
| Native 구현 검증 | 1회 | 첫 2태스크, 각 2epoch | 66 |
| 독립 저자 코드 기반 검증 | PEFT 7방법의 각 조건 1회 | 첫 2태스크, 각 2epoch | 42 |
| Native 성능·효율성 본 측정 | 3회 | 첫 2태스크, 각 20epoch | 198 |
| 별도 지원 연산 FLOPs | 1회 | 첫 2태스크, 각 20epoch | 66 |
| 합계 | | | 372 |

42회는 원 저자 소스를 고정하고 크기·공통 입력을 명시적으로 조정한 독립 기준선이다. 원 논문의 Tiny/Small 성능 재현으로 해석하지 않는다. 일반 CL 4방법에는 이 공통 ViT recipe의 공식 저자 기준선이 없으며, 해당 24 native 검증은 이미 66회 안에 포함되어 있다. 일반 CL에는 학습 정상성·업데이트·프레임워크 계약을 검증하고 저자 정확도와의 1.5/3%p 판정을 주장하지 않는다.

지원되는 크기 순서 Tiny→Small→Base, 각 크기 batch16/64에서 검증한다. 해당 조건의 검증 통과 후 측정/FLOPs로 넘어간다. 같은 seed의 3회는 비용 변동을 측정하며 성능의 seed 간 불확실성을 추정하지 않는다. 정확도와 자원 통계에는 본 측정 198회만 사용한다. 검증·기준선·FLOPs 실행의 자원 비용은 별도 기록한다.

기준선과 native의 입력·초기화·recipe가 일치할 때 정확도 하락 ≤1.5%p 통과, 1.5–3.0%p 추가 점검, >3.0%p 실패. 비정상적 저성능·NaN·실제 업데이트 누락은 조사하고, 해결되지 않은 조건은 본 측정을 진행하지 않는다. Tiny와 Base의 절대 정확도 차이나 정상적인 Fine-tuning의 망각을 구현 오류로 판정하지 않는다.

## 명목 epoch 외 비용

- HiDe-Prompt: 검증은 태스크마다 TII2+prompt2, 본 측정/FLOPs는 TII20+prompt20. 두 경로의 통계 수집과 태스크2의 각 classifier alignment30도 포함한다.
- RanPAC: 태스크1 adapter 학습, 태스크2 optimizer update0. 실제 남아 있는 loader 방문과 특징 추출, random projection, ridge 선택·해결을 포함한다.
- EWC-LoRA: prototype, empirical Fisher, 태스크2 EWC penalty, 평가 후 merge/reset 등 finalize를 포함한다.
- InfLoRA: 시작 시 activation/SVD, 종료 시 prototype/DualGPM, 태스크2 부분공간 제약을 포함한다.
- CL-LoRA: 태스크2의 KD→CE 두 optimizer update, prototype 및 shared adapter snapshot을 포함한다.
- CODA-Prompt: query/prompt forward, 태스크2 prompt 재구성과 optimizer 재생성을 포함한다.
- EASE: prototype 추출·보완, 태스크2 확장된 특징·추론 비용을 포함한다.
- LwF-MC: teacher copy/storage 및 태스크2 teacher forward를 포함한다.

본 비교는 동일 데이터·명목 epoch의 공통 비교이며 동일 FLOPs/energy 예산 비교가 아니다. 방법별 추가 단계와 실제 업데이트 횟수는 유지하고 함께 보고한다. 첫 2태스크 결과로 장기 망각이나 장기 모델 증가의 우위를 확정하지 않는다.

## 계측과 저장

초기화·데이터 로딩·태스크 시작·학습·종료·평가·통계/Fisher/SVD/CA/ridge·로컬 checkpoint 비용을 기록한다. 전체 wall time과 구간별 시간, CPU time/RSS, CUDA allocated/reserved peak, NVML device memory, GPU power raw samples와 energy/coverage, 실제 forward/backward/update/image visit, 파라미터·teacher·buffer·통계 등 중복 제외 저장량을 수집한다. 겹치는 구간은 중복 합산하지 않는다. GPU 보드 에너지는 전체 머신 에너지가 아니며 지원 연산 FLOPs는 전체 연산량의 완전한 값이 아니다.

평상시 태스크 평가의 시간·정확도를 함께 기록한다. 별도 추론 throughput sweep은 이번 범위에 포함하지 않는다. 결과는 `results/unified_comparison` 아래 JSON/CSV·raw 계측·보고서와 W&B `laymond1/cl-efficiency`에 저장한다. Slack은 시작·진행·실패·완료를 알린다. 기존 환경변수/로컬 설정의 비밀값은 계획·결과·로그에 저장하지 않는다. 원격 업로드 비용은 학습 효율성 통계에서 분리한다.

## 이번 범위에서 제외

Online EWC, SI, L2P, DualPrompt, RainbowPrompt, SD-LoRA를 이번 재측정에서 제외한다. 구현과 기존 결과는 보존한다. PROL/LW2G 및 BudgetCL/aL-SAR 비교, 전체 10태스크, 추가 seed, 비율 정렬·rank/prompt sweep, 자동 retry/recipe 변경, 원격 git push도 이번 범위에 포함하지 않는다.

## 실행 전 남은 준비

기존 `protocol.json`과 `scripts/unified_cl_protocol.py`는 아직 17방법의 이전 초안이다. 직접 실행 기준으로 사용하지 않는다. 최종 확인 후 11방법·GPU0/1·평가 batch64·cuDNN TF32true 설정으로 동기화하고 통합 runner/launcher 및 독립 기준선 연결을 마친다. 공통 CPU helper는 검사했으나 이 검토안의 GPU 검증/본 측정은 아직 실행하지 않았다. 독립 기준선 42회는 완료된 queue가 아닌 준비할 계획이다.

GPU 교차 점검을 별도 실행한다면 위 372회와 구분해서 기록한다. 마지막 확인안에는 EASE Tiny/batch64 본 측정 1회를 다른 GPU에서 추가하고 기존 본 측정 1회와 대조하는 계획을 제안한다. 추가 계측은 기존 198회 통계에 섞지 않으며, 이를 포함한 계획 총량은 373회이다. 비교 조건·실제 센서 지원을 확인한 뒤 구체적인 배치 위치를 정한다.
