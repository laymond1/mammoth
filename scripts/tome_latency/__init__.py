"""ToMe batch-1 latency re-check: eager grid, kernel attribution, compile and CUDA graph (stages 1 and 2).

Reuses ``scripts.token_efficiency`` for the model, the timing loop and the kernel profiler, so the numbers
stay comparable to the five-device study. Nothing here changes a token schedule or an operator.
"""
