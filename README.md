# MedConstBench

**Benchmarking How Constitutions Steer LLMs in Medical Ethics**

MedConstBench measures how a deployment-time *ethical constitution* (an ethics guideline injected
as a system prompt) changes the behavior of large language models on medical-ethics questions,
scored along **Beauchamp's four principles** (autonomy, non-maleficence, beneficence, justice).

## Layout

```
constitutions/         # the four constitutions, one text file each (C0 None / C1 AMA / C2 WHO / C3 Mitigation)
src/constitutions.py   # loads the four constitution files
src/config.py          # model list, 12-item rubric, chat-message builder
src/run_inference.py   # generate answers under each constitution (vLLM)
src/run_judge.py       # multi-model mutual evaluation (vLLM)
src/aggregate.py       # judgments -> per-principle scores
data/benchmark_400.json        # the 400-question benchmark (200 MedEthicsQA + 200 MedDilemma)
data/medethics_open_questions.json  # referenced MedEthicsQA question pool
scripts/run_all.sh     # end-to-end pipeline
outputs/               # results are written here
```

## Dataset

`data/benchmark_400.json` has two subsets, organized by the four principles: **MedEthicsQA**
(open-ended discourse questions, sampled from
[MedEthicsQA](https://github.com/JianhuiWei7/MedEthicsQA)) and **MedDilemma** (clinical dilemmas
authored for this work, each staging a conflict between two principles, covering all six pairwise
conflicts). MedDilemma items were drafted with an LLM and curated by the authors; the generator is
recorded in `source` and is not among the evaluated models.

## Usage

```bash
pip install -r requirements.txt          # numpy; vLLM for GPU inference
bash scripts/run_all.sh smoke            # quick sanity check
bash scripts/run_all.sh full             # full run (GPU; set model paths in src/config.py)
```

Run from the repository root. Model paths in `src/config.py` point to a local cluster; edit them
to your own paths or Hugging Face repo IDs.

Discourse items are derived from [MedEthicsQA](https://github.com/JianhuiWei7/MedEthicsQA);
please cite the original work if you use them.

## Citation

If you use MedConstBench, please cite:

```bibtex
@article{yang2026medconstbench,
  title={MedConstBench: Benchmarking How Constitutions Steer LLMs in Medical Ethics},
  author={Yang, Yibo and Huang, Zerui and Yang, Haomiao and Wang, Zixia},
  journal={arXiv preprint arXiv:2610.36513},
  year={2026}
}
```
