# Risk-Concealment Attacks (RCA)

Official codebase for the paper: **Uncovering the Vulnerability of Large Language Models in the Financial Domain via Risk Concealment**

**TL;DR:** We propose **Risk-Concealment Attacks (RCA)** - a multi-turn red-teaming framework that adaptively conceals malicious financial intent across dialogue turns to elicit high-risk outputs from LLMs, even under strict alignment. RCA achieves an average attack success rate (ASR) of **93.18%** across 9 mainstream LLMs, including **98.28% on GPT-4.1** and **97.56% on OpenAI o1**, revealing critical gaps in financial LLM safety.


## 📌 Overview
Large Language Models (LLMs) are increasingly deployed in high-stakes financial applications. While current alignment techniques effectively block explicit harmful queries (e.g., violence, hate speech), they often fail to detect **regulatory gray zones** such as tax evasion, insider trading, or market manipulation when phrased in a seemingly compliant way.

To evaluate this vulnerability, we introduce:

- **🧠 RCA (Risk-Concealment Attacks):** An adaptive, multi-turn attack framework that starts with a benign-looking financial query and incrementally injects risk across follow-up turns.
- **📊 FIN-Bench:** A benchmark of 522 harmful financial queries categorized into 6 real-world regulatory risk types.
- **⚖️ Comprehensive Evaluation:** Extensive experiments on 9 leading LLMs (e.g., GPT-4.1, OpenAI o1, Claude Sonnet 3.7, Claude Sonnet 4, Qwen3 235B) demonstrating the effectiveness and transferability of RCA.

### RCA (Risk-Concealment Attacks)
![rca_framework](assets/example.png) <div align="center">Figure 1. Comparison between explicit and implicit risks in prompt inputs. </div>

As illustrated in Figure 1, LLMs demonstrate strong safeguards against explicit harmful prompts, such as requests related to suicide or bomb-making, consistently responding with firm refusals. However, when confronted with prompts situated in regulatory gray areas, such as tax minimization through offshore entities, the model often fails to issue a rejection and instead produces helpful responses. This highlights a critical vulnerability: prompts that maintain a facade of legality or professional tone may evade detection, even when they encode high-risk financial intent that could violate regulatory norms.

![rca_framework](assets/rca_framework.png) <div align="center">Figure 2. Overview of RCA framework. </div>

The RCA framework consists of two phases. In **Phase 1**, we construct an initial prompt using a structured template that encodes risky financial intent. In **Phase 2**, we iteratively refine follow-up queries based on feedback, such as jailbreak indicators and dialogue history from the Judge agent. This adaptive process reduces perceived risk while gradually eliciting actionable, compliance-violating responses from the target LLM.


### FIN-Bench
 We constructed a comprehensive dataset of 522 harmful financial queries. The dataset sources include:
 * 240 queries collected from existing red-teaming benchmarks such as [AdvBench](https://github.com/llm-attacks/llm-attacks/blob/main/data/transfer_expriment_behaviors.csv), [AART](https://github.com/google-research-datasets/aart-ai-safety-dataset), [Beavertails](https://github.com/PKU-Alignment/beavertails), [Do-Not-Answer](https://github.com/haizelabs/redteaming-resistance-benchmark/tree/26dc0592e2879c6cb4d1627500fb5d3cc097df5d/data/do-not-answer), and [RED-EVAL](https://github.com/declare-lab/red-instruct)
* 240 GPT-generated prompts (using GPT-4o) simulating illegal or high-risk financial queries
* 42 semantically rewritten prompts based on user-facing questions from major financial institutions' FAQ sections, adversarially reframed with malicious financial intent. This multi-source curation enhances realism, linguistic diversity, and scenario complexity.

We provide category descriptions in FIN-Bench, which contain six financial behaviors, including Financial Fraud, Insider Trading, Market Manipulation, Money Laundering, Regulatory Circumvention, and Tax Evasion.

* **Financial Fraud**. Deceptive practices intended for financial gain, including false statements, forged documents, fraudulent investment schemes, misleading marketing, and misappropriation of funds.
* **Insider Trading**. The use of material non-public information (MNPI) to gain an unfair advantage in securities trading before the information becomes public.
* **Market Manipulation**. The artificial distortion of a financial asset’s price or liquidity through tactics such as coordinated trading, false information, spoofing, or pump-and-dump schemes.
* **Money Laundering**. The process of concealing the origins of illicit funds through techniques such as structuring, layering, asset transfers, and the use of shell companies to make the money appear legitimate.
* **Regulatory Circumvention**. The strategic structuring of products, contracts, or workflows to formally comply with regulations while avoiding substantive regulatory obligations, such as disclosure, classification, or risk management.
* **Tax Evasion**. The illegal reduction of tax liability by concealing income, misreporting information, using offshore accounts, or exploiting regulatory loopholes.

To provide a comprehensive view of the dataset composition, we show the distribution of data sources in Figure 3, and the proportion of each data source within each financial category in Figure 4.

<p align="center">
<img src="assets/data_source_distribution.png" width="480" height="380"/>
</p>
<div align="center">Figure 3. Distribution of data source in FIN-Bench</div> </br>

<p align="center">
<img src="assets/proportion.png"/>
</p>
<div align="center">Figure 4. Proportion of data sources for each financial behav-
ior category.</div>

## 📈 Main Result
We report the ASR of various jailbreak methods against 9 popular aligned LLMs in Table 1, covering both single-turn and multi-turn settings. We further demonstrate that RCA consistently achieves high success rates while evading safety filters.

 | Method        | LLaMA 3.3 70B | Qwen2.5 72B | Gemini 2.5 Flash | GPT-4o | Qwen3 235B | GPT-4.1 | Claude Sonnet 3.7 | o1   | Claude Sonnet 4 | Avg   |
|---------------|---------------|-------------|------------------|--------|------------|---------|-------------------|------|-----------------|-------|
| **Single-Turn Attack Method** ||||||||||||
| ArtPrompt     | 40.91         | 39.13       | 13.04            | 36.36  | 59.09      | 43.48   | 8.70              | 0.00 | 0.38            | 26.78 |
| ReNeLLM       | 85.19       | 77.27       | 94.24          | 90.18| 95.24    | 93.94   | 70.77             | 31.64| 7.04            | 71.72 |
| DeepInception | 70.00         | 36.25       | 58.75            | 42.53  | 23.53      | 61.25   | 9.50              | 5.00 | 0.00            | 34.09 |
| FlipAttack    | 42.94         | 93.87     | 92.64            | 82.72  | 92.25      | 95.09 | 81.82             | 0.77 | 0.19            | 64.69 |
| CodeChameleon | 67.42         | 65.81       | 84.09            | 74.05  | 89.15      | 93.18   | **89.30**         | 6.80 | **77.78**       | 71.95 |
| **Multi-Turn Attack Method** ||||||||||||
| Crescendo     | 60.42         | 36.73       | 30.00            | 28.00  | 46.00      | 28.57   | 6.12              | 8.00 | 0.00            | 27.09 |
| FITD          | 62.00         | 72.00       | 66.00            | 77.17  | 84.62      | 64.00   | 75.51             | 79.59 | 69.10        | 72.22 |
| Ours          | **97.70**     | **98.47**   | **95.25**        | **97.51**| **97.50** | **98.28** | 85.20         | **97.56** | 71.20   | **93.18** |
<div align="center">Table 1: Comparison between our framework and competing methods across nine LLMs by attack success rate (%). We bolded the maximum number and underlined the second-highest. All multi-turn attack methods allowed a maximum of five rounds.</div> </br>

To analyze the efficiency between our framework and competing methods, we report input token, average runtime, and ASR in Table 2. The results indicate that our method achieves the highest ASR (97.51%) while maintaining one of the lowest runtimes (6.59s), outperforming all other methods.
<div align="center">

| Method        | Input Tokens | Avg. Time (s) | ASR (%) |
|---------------|--------------|---------------|---------|
| ArtPrompt     | 1,831        | 13.94         | 36.36   |
| ReNeLLM       | 7,217        | 27.69         | 90.18   |
| DeepInception | **564**      | 8.94          | 42.53   |
| FlipAttack    | 2,579        | 6.77          | 82.72   |
| CodeChameleon | 5,524        | 6.99          | 74.05   |
| Crescendo     | 52,621       | 30.92         | 28.57   |
| FITD          | 59,366       | 59.00         | 77.17   |
| RCA           | 9,025        | **6.59**      | **97.51** |

</div>
<div align="center">Table 2: Analysis of efficiency by average token usage, average runtime per successful attack (seconds), and ASR (%). </div> </br>

Figure 5 reports the jailbreak success rate (ASR) of our method on GPT-4o across six financial behavior categories.
<p align="center">
<img src="assets/ars_radar.png" width="580" height="380"/>
</p>
<div align="center">Figure 5. Breakdown of attack success by categories.</div> </br>

Figure 6 shows that in the first round, over 21\% of the generated questions still retain high-risk signals. However, after only two refinement rounds, high-risk prompts vanish entirely, and low-risk prompts increase steadily, reaching 40\% by round 5. This supports our hypothesis that Iterative Deception via Contextual Refinement helps progressively mask malicious financial intent through semantically and stylistically aligned rewrites.
<p align="center">
<img src="assets/risk_distribution.png"/>
</p>
<div align="center">Figure 6. Risk level distribution and ASR across RCA refine-
ment rounds. RCA reduces risk while increasing ASR.</div>

## 🚀 Quick Start
To reproduce the results on **FIN-Bench**, you need to provide:
- API access to the target LLMs (e.g., OpenAI, Anthropic, Google).

### Install the environment
```pip install -r requirements.txt```

### Run RCA Experiments
To calculate ASR of RCA on FIN-Bench. Set up your target LLM models and related API key.
```bash
python rca.py \
  --dataset_path ./data/finbench.csv \
  --target_llm_models gpt-4.1 o1 claude-sonnet-4 \
  --auxilia_model llama-3.1-70b \
  --judgement_model gpt-4.1 \
  --max_num_rounds 5 \
  --temperature 0.01
```
Output

* Logs will be shown in the console.
* Per-model experiment results are saved to CSV files in the results/ directory, e.g.: rca_gpt-4.1_num_round.csv, rca_gpt-4.1_label.csv. Each CSV file contains: num_round / label,success_rate, total_cases

## 📚 Citation

If you find this repository helpful, please cite our paper.

```bibtex
@article{anonymous2025rca,
  title     = {Uncovering the Vulnerability of Large Language Models in the Financial Domain via Risk Concealment},
  author    = {Anonymous},
  journal   = {Under Review},
  year      = {2025}
}
```


## 🔐 Disclaimer

This code is intended for **research and security analysis only**. The prompts and queries in this repository simulate adversarial intent but should **never be used to promote or deploy harmful content**.

We release this framework to encourage responsible red-teaming and LLM safety research in high-stakes domains.
