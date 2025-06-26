# Evaluation Tools

Scripts for assessing generated volumes and performing statistical analysis. The evaluation procedure is summarized in [../Resource/Main1.pdf](../Resource/Main1.pdf).

## Scripts
- `1_total_segment_gen_population.py` – run TotalSegmentator across the generated population.
- `2_Data_AutoPET.py` – AutoPET data loader for evaluation.
- `3_stat_all_and Tests_MF-anova.py` – statistical tests for multiple folds.

## Workflow
1. Segment the generated volumes using `1_total_segment_gen_population.py`.
2. Load data with `2_Data_AutoPET.py` to compute metrics.
3. Run `3_stat_all_and Tests_MF-anova.py` to produce statistical summaries.

These routines were tested locally. Key results are reported in [../Resource/BMIs.pdf](../Resource/BMIs.pdf) and [../Resource/Graph.pdf](../Resource/Graph.pdf). Consult the [arXiv version](https://arxiv.org/pdf/2505.22489) for further details.
