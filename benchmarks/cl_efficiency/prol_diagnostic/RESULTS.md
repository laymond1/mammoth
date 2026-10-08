# PROL checkpoint 추론 진단

저장된 첫 태스크 checkpoint에서 새로 복원한 저자 모델 A와 compact clone B를 비교했습니다. 각 모델에서 audit ON/OFF를 10회씩 평가하여 **40회 모델 평가·80회 encoder forward·20개 A/B 쌍**을 관측했습니다.

입력은 기존 평가 마지막 **40개 이미지**이며 동일 이미지를 반복하여 1,600번 방문했습니다. 학습·backward·optimizer update는 모두 0회입니다. 별도의 CL 정확도나 효율성 pilot 결과가 아닙니다.

| 비교 | 전체 쌍 | bitexact 불일치 | logit 최대 절대 차이 |
|---|---:|---:|---:|
| A/B | 20 | 20 | 0.193215609 |
| 같은 조건 반복 | 180 | 180 | 0.193215609 |
| audit ON/OFF | 20 | 20 | 0.17527318 |

관측 분류: **reference_clone_output_difference_observed**.

**이번 행렬에서 재현되지 않아도 원인이 해결되었다고 판정하지 않습니다.** 원래 학습 프로세스의 probe logits/RNG가 저장되지 않았으므로 당시 live 객체를 그대로 재현할 수 없습니다. audit ON은 원래 hook 연산에 내부 trace와 CPU 복사를 추가해 실행 순서에 영향을 줄 수 있습니다. hook과 출력 변동의 관계는 CUDA 중복 쓰기 승자의 인과적 증명이 아닙니다.

저장된 기존 40개 예측과의 일치 수는 진단 자료이며 새 태스크 정확도가 아닙니다. 기존 checkpoint 검사와 구현 통과 판정은 변경하지 않았고, 후속 학습·재시도는 승인되지 않았습니다.

[원래 실패한 저자 실행](https://wandb.ai/laymond1/cl-efficiency/runs/2qvxzkht) · [이번 추론 진단](https://wandb.ai/laymond1/cl-efficiency/runs/v80kh0np)
