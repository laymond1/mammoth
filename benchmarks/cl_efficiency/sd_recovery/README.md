# SD-LoRA recovery stage

The approved scope was completed on GPU 1: one native first-task validation, three fresh cost measurements and one separate supported-operator FLOPs pass. The saved independent author result was reused without retraining. Native and author accuracy were both 99.0%; final network hashes and predictions matched.

SD uses common crop min 0.05, encoder/scale LR 0.001 and classifier LR 0.008. Model defaults remain 0.008/0.008. Existing eight-method results, the failed 10% observation and the original deferred jobs remain intact. Three same-seed SD measurements are combined with the prior eight accepted methods; validation, diagnostic and profiler resource costs are excluded from means.

See RESULTS.md, results.csv and efficiency_comparison.png/pdf. checks.json records actual validation commands. publication.json, source_readback.json and publication_readback.json record W&B and Slack completion after remote verification. Local author diagnostics are archived as existing evidence; their original local-only metadata is preserved.

This stage covers only CIFAR100 first task / one epoch, ViT-B/16 IN21K→IN1K, batch64, rank10, seed1993, workers4, FP32. Energy is GPU-board energy and FLOPs are partial supported-operator estimates. This does not establish full-sequence forgetting, multi-seed quality or equal-budget comparisons. EASE CPU-thread metadata remains unrecorded.

Frozen training commit: bdad261. No model/config/source changes, LR search, author retraining, second-task training, full ten-task training or remote Git push were performed in this stage.
