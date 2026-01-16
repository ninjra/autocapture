# Prompt Strategies (Repetition + Zero-shot CoT)

Autocapture supports prompt-level strategies that can improve non-reasoning accuracy
and optional reasoning performance without changing downstream workflows. Strategies
are applied at the LLM boundary and logged with structured metadata.

## Prompt repetition (non-reasoning boost)

Prompt repetition duplicates the final user prompt text (with a delimiter) so the
model can attend to the full prompt more effectively. This is based on the findings
in *Prompt Repetition Improves Non-Reasoning LLMs* (arXiv:2512.14982).

**Tradeoffs**
* Increases input tokens (prefill latency), especially for long prompts.
* Automatically degrades to avoid exceeding context limits.

## Step-by-step mode (Zero-shot CoT)

Zero-shot chain-of-thought adds the phrase **"Let's think step by step."** to
encourage multi-step reasoning (Kojima et al., arXiv:2205.11916). This mode is
optional and can be combined with repetition when safe.

**Tradeoffs**
* Increases output tokens and latency.
* Two-stage mode keeps rationales internal and returns final answers only.

## Configuration

```yaml
llm:
  prompt_strategy_default: "repeat_2x"
  prompt_repeat_factor: 2
  enable_step_by_step: false
  step_by_step_phrase: "Let's think step by step."
  step_by_step_two_stage: false
  max_prompt_chars_for_repetition: 12000
  max_tokens_headroom: 512
  max_context_tokens: 8192
  force_no_reasoning: false
  strategy_auto_mode: true
  prompt_repetition_delimiter: "\n\n---\n\n"
  store_prompt_transforms: false
  prompt_store_redaction: true
```

### Recommended defaults

* **Non-reasoning QA / summarization** → `repeat_2x` with step-by-step **off**.
* **Reasoning-heavy tasks** → enable step-by-step (and optional two-stage) only
  when latency is acceptable.

## PromptOps A/B evals

Run the offline eval harness to compare strategies without calling external APIs:

```bash
python -m promptops.eval --strategies baseline,repeat2,repeat3,step,step+repeat2
```

Results are stored as JSONL with per-case metrics and prompt strategy metadata.

To run against live providers, set `RUN_LIVE_EVALS=1` and pass `--live`.
