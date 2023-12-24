# kd_early_ranker
Covers a couple of approaches to training an early ranker with knowledge distillation from final ranker.

Files:
1. baseline_early_ranker.py : Shows how normally early ranker is trained.
1. kd_aux_early_ranker.py : Shows how we can train using auxiliary tasks corresponding to teacher labels.
1. kd_shared_early_ranker.py : Uses the shared logits approach to KD.
