# ViT 크기와 adapter 비교 준비

세 크기를 AugReg **IN21K → IN1K fine-tuned** 계열로 맞춥니다. 같은 weight family라도 크기별 pretraining recipe·학습 계산량까지 동일하다는 뜻은 아닙니다. 파일/hash와 weight metadata를 실행 결과에 기록해야 하며, 현재는 registry 확인만 했습니다.

| 설정 | timm tag | feature D | adapter r | adapter parameters/task 예상 |
|---|---|---:|---:|---:|
| Base 공통64 | vit_base_patch16_224.augreg_in21k_ft_in1k | 768 | 64 | 1,189,632 |
| Small 공통64 | vit_small_patch16_224.augreg_in21k_ft_in1k | 384 | 64 | 595,200 |
| Tiny 공통64 | vit_tiny_patch16_224.augreg_in21k_ft_in1k | 192 | 64 | 297,984 |
| Small 비율32 | 위 Small tag | 384 | 32 | 299,904 |
| Tiny 비율16 | 위 Tiny tag | 192 | 16 | 76,224 |

위 수치는 12 layers × `(2Dr + r + D)`의 구조상 계산으로, 실험에서 측정한 결과가 아닙니다. Frozen backbone, proxy head, prototype buffers와 optimizer state는 제외합니다. Base64는 두 비교군에서 동일하므로 다섯 개 고유 설정만 준비했습니다.

공통64 비교는 bottleneck 절대 크기를 유지합니다. 상대 용량은 Tiny가 크므로, 추가 Tiny16·Small32·Base64 비교에서 `r/D=1/12`를 유지합니다. 두 비교군의 결과를 섞어 크기 효과와 adapter 용량 효과를 해석하지 않습니다.

작은 ViT는 일반적으로 backbone 저장량과 계산량을 줄이는 선택이지만, 실제 정확도·처리량·전력/에너지 이득은 아직 측정하지 않았습니다. EASE는 태스크마다 adapter를 추가하고 모든 관측 adapter 공간으로 추론합니다. 따라서 첫 태스크 수치만으로 여러 태스크의 비용을 추정하지 말고, 추후 task별 parameter/buffer bytes와 prototype/evaluation phase를 함께 확인해야 합니다.

모델과 parser는 크기별 D·attention head 수를 timm에서 얻습니다. Adapter width, proxy head, prototype concatenation과 checkpoint 검증도 D를 따라가며, Base의 768을 하드코딩하지 않았습니다. 학습 없는 metadata tests에서 세 크기의 선택과 상태 구성을 검증했습니다. 실제 Tiny·Small forward/backward와 학습 정확도는 아직 미검증입니다.

이제 추가 사용자 결정은 필요하지 않습니다. 다음 실제 실행 전에 checkpoint hash와 공식 코드 대비 허용 accuracy 차이를 확정하고, 원칙대로 Base부터 검증합니다. 이 항목은 [실행 계획](validation_plan.json)의 선행 조건으로 남겼습니다.
