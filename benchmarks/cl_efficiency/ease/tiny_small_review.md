# ViT 크기와 adapter 비교 준비

세 크기를 AugReg **IN21K → IN1K fine-tuned** 계열로 맞춥니다. 같은 weight family라도 크기별 pretraining recipe·학습 계산량까지 동일하다는 뜻은 아닙니다. 파일/hash와 weight metadata를 실행 결과에 기록해야 하며, Base의 파일/hash는 실제 smoke·공식 reference·native 두 태스크 검증에서 확인했습니다. Base64·seed1993은 네 정확도 지표와 일곱 구조 검사를 모두 통과했습니다. [Native 검증 보고서](native_validation_report.md)와 [JSON 결과](native_validation_results.json)를 참고합니다. Tiny·Small은 registry 확인만 했으며 파일/hash와 실제 forward/backward는 아직 미검증입니다.

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

Tiny·Small의 family와 adapter 설정은 확정됐습니다. Base64·seed1993의 두 태스크 acceptance 평가는 공식 기준선 대비 정확도 하락 1.5%p 이내 조건을 통과했습니다. 다음 크기 실험은 checkpoint 파일/hash와 실행 범위를 정한 뒤 진행합니다. Tiny·Small 학습, 공통 효율성 pilot, FLOPs 측정, 병합과 push는 아직 수행하지 않았습니다. 상세 기준은 [실행 계획](validation_plan.json)에 기록했습니다.
