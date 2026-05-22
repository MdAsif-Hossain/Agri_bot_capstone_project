# Papers to Add — AgriBot IEEE Conference Paper

> **Current count:** 16 papers reviewed  
> **After adding these:** ~35 papers (IEEE-ready)  
> **Rule:** Tier 1 papers are non-negotiable. A reviewer who sees FAISS + BM25 + RRF + Whisper + MobileNetV3 + BanglaT5 without their foundational citations will reject immediately.

---

## 🔴 Tier 1 — Absolutely Required (11 papers)

### 1. Original RAG Paper
**Lewis et al., NeurIPS 2020**

| Field | Value |
|---|---|
| Title | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks |
| Authors | Lewis, Perez, Piktus, Petroni, Karpukhin, Goyal, Küttler, Lewis, Yih, Rocktäschel, Riedel, Kiela |
| Venue | NeurIPS 2020, pp. 9459–9474 |
| Why | You built a RAG system. This is citation #1. Every reviewer expects it. |

```bibtex
@inproceedings{Lewis2020RAG,
  author    = {Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and
               Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and
               K{\"u}ttler, Heinrich and Lewis, Mike and Yih, Wen-tau and
               Rockt{\"a}schel, Tim and Riedel, Sebastian and Kiela, Douwe},
  title     = {Retrieval-Augmented Generation for Knowledge-Intensive {NLP} Tasks},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {33},
  pages     = {9459--9474},
  year      = {2020}
}
```

---

### 2. FAISS — Dense Vector Search
**Johnson et al., IEEE Transactions on Big Data 2021**

| Field | Value |
|---|---|
| Title | Billion-Scale Similarity Search with GPUs |
| Authors | Johnson, Douze, Jégou (Meta AI Research) |
| Venue | IEEE Transactions on Big Data, Vol. 7, No. 3, pp. 535–547 |
| DOI | 10.1109/TBDATA.2019.2921572 |
| Why | Your entire dense retrieval pipeline uses FAISS IndexFlatIP. |

```bibtex
@article{Johnson2021FAISS,
  author    = {Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  title     = {Billion-Scale Similarity Search with {GPUs}},
  journal   = {IEEE Transactions on Big Data},
  volume    = {7},
  number    = {3},
  pages     = {535--547},
  year      = {2021},
  doi       = {10.1109/TBDATA.2019.2921572}
}
```

---

### 3. BM25 — Sparse Retrieval
**Robertson & Zaragoza, Foundations and Trends in IR 2009**

| Field | Value |
|---|---|
| Title | The Probabilistic Relevance Framework: BM25 and Beyond |
| Authors | Robertson, Zaragoza |
| Venue | Foundations and Trends in Information Retrieval, Vol. 3, No. 4, pp. 333–389 |
| DOI | 10.1561/1500000019 |
| Why | Your BM25Okapi sparse retriever is directly based on this framework. |

```bibtex
@incollection{Robertson2009BM25,
  author    = {Robertson, Stephen and Zaragoza, Hugo},
  title     = {The Probabilistic Relevance Framework: {BM25} and Beyond},
  booktitle = {Foundations and Trends in Information Retrieval},
  volume    = {3},
  number    = {4},
  pages     = {333--389},
  year      = {2009},
  publisher = {Now Publishers},
  doi       = {10.1561/1500000019}
}
```

---

### 4. Reciprocal Rank Fusion (RRF) — Your Hybrid Retrieval Core
**Cormack, Clarke, Büttcher — SIGIR 2009**

| Field | Value |
|---|---|
| Title | Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods |
| Authors | Cormack, Clarke, Büttcher |
| Venue | SIGIR 2009, pp. 758–759 |
| DOI | 10.1145/1571941.1572114 |
| Why | Your `_reciprocal_rank_fusion()` function in `hybrid.py` implements this exact algorithm. |

```bibtex
@inproceedings{Cormack2009RRF,
  author    = {Cormack, Gordon V. and Clarke, Charles L. A. and B{\"u}ttcher, Stefan},
  title     = {Reciprocal Rank Fusion Outperforms Condorcet and Individual
               Rank Learning Methods},
  booktitle = {Proceedings of the 32nd International ACM SIGIR Conference on
               Research and Development in Information Retrieval},
  pages     = {758--759},
  year      = {2009},
  doi       = {10.1145/1571941.1572114}
}
```

---

### 5. Sentence-BERT — Embedding Model Lineage
**Reimers & Gurevych — EMNLP 2019**

| Field | Value |
|---|---|
| Title | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks |
| Authors | Reimers, Gurevych |
| Venue | EMNLP 2019, pp. 3982–3992 |
| DOI | 10.18653/v1/D19-1410 |
| Why | Your `all-MiniLM-L6-v2` embedding model is from the Sentence-Transformers family introduced here. |

```bibtex
@inproceedings{Reimers2019SBERT,
  author    = {Reimers, Nils and Gurevych, Iryna},
  title     = {Sentence-{BERT}: Sentence Embeddings using Siamese {BERT}-Networks},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in
               Natural Language Processing (EMNLP)},
  pages     = {3982--3992},
  year      = {2019},
  doi       = {10.18653/v1/D19-1410}
}
```

---

### 6. Whisper ASR — Your Primary STT Engine
**Radford et al. — ICML 2023**

| Field | Value |
|---|---|
| Title | Robust Speech Recognition via Large-Scale Weak Supervision |
| Authors | Radford, Kim, Xu, Brockman, McLeavey, Sutskever (OpenAI) |
| Venue | ICML 2023, pp. 28492–28518 |
| Why | Your `faster-whisper` STT module is built on this model. Used across all voice endpoints. |

```bibtex
@inproceedings{Radford2023Whisper,
  author    = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg
               and McLeavey, Christine and Sutskever, Ilya},
  title     = {Robust Speech Recognition via Large-Scale Weak Supervision},
  booktitle = {Proceedings of the 40th International Conference on Machine
               Learning (ICML)},
  pages     = {28492--28518},
  year      = {2023}
}
```

---

### 7. MobileNetV3 — Your Vision Classifier Backbone
**Howard et al. — ICCV 2019**

| Field | Value |
|---|---|
| Title | Searching for MobileNetV3 |
| Authors | Howard, Sandler, Chu, Chen, Chen, Tan, Wang, Zhu, Pang, Vasudevan, Le, Adam (Google) |
| Venue | ICCV 2019, pp. 1314–1324 |
| DOI | 10.1109/ICCV.2019.00140 |
| Why | Your `classifier.py` and `PROJECT_PHASES_COMPLETE_BREAKDOWN.txt` explicitly name MobileNetV3 as the crop disease CNN. |

```bibtex
@inproceedings{Howard2019MobileNetV3,
  author    = {Howard, Andrew and Sandler, Mark and Chu, Grace and Chen,
               Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and
               Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and
               Le, Quoc V. and Adam, Hartwig},
  title     = {Searching for {MobileNetV3}},
  booktitle = {Proceedings of the IEEE/CVF International Conference on
               Computer Vision (ICCV)},
  pages     = {1314--1324},
  year      = {2019},
  doi       = {10.1109/ICCV.2019.00140}
}
```

---

### 8. CBAM — Your CNN Attention Module
**Woo et al. — ECCV 2018**

| Field | Value |
|---|---|
| Title | CBAM: Convolutional Block Attention Module |
| Authors | Woo, Park, Lee, Kweon |
| Venue | ECCV 2018, pp. 3–19 |
| DOI | 10.1007/978-3-030-01234-2_1 |
| Why | Your vision classifier uses `MobileNetV3 + CBAM` — CBAM is explicitly named in your system docs and `classifier.py`. |

```bibtex
@inproceedings{Woo2018CBAM,
  author    = {Woo, Sanghyun and Park, Jongchan and Lee, Joon-Young and
               Kweon, In So},
  title     = {{CBAM}: Convolutional Block Attention Module},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  pages     = {3--19},
  year      = {2018},
  doi       = {10.1007/978-3-030-01234-2_1}
}
```

---

### 9. Qwen2.5 — Your LLM
**Qwen Team, Alibaba — arXiv 2025**

| Field | Value |
|---|---|
| Title | Qwen2.5 Technical Report |
| Authors | Qwen Team, Alibaba Group (An Yang, Baosong Yang, et al.) |
| arXiv | 2412.15115 |
| DOI | 10.48550/arXiv.2412.15115 |
| Why | `qwen2.5-1.5b-instruct-q8_0.gguf` is your generation engine. Reviewers will ask for a citation. |

```bibtex
@misc{Qwen2025,
  author        = {{Qwen Team}},
  title         = {{Qwen2.5} Technical Report},
  year          = {2025},
  eprint        = {2412.15115},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2412.15115},
  doi           = {10.48550/arXiv.2412.15115}
}
```

---

### 10. BanglaT5 — Your Translation Model (Two Citations)
**Bhattacharjee et al. — EACL 2023 + Hasan et al. — EMNLP 2020**

| Field | Value |
|---|---|
| Title (model) | BanglaNLG and BanglaT5: Benchmarks and Resources for Evaluating Low-Resource NLG in Bangla |
| Authors | Bhattacharjee, Hasan, Ahmad, Shahriyar (BUET CSE NLP Group) |
| Venue | Findings of EACL 2023 |
| Title (corpus) | Not Low-Resource Anymore: Aligner Ensembling, Batch Filtering, and New Datasets for Bengali-English MT |
| Authors | Hasan, Bhattacharjee, Samin, Hasan, Basak, Rahman, Shahriyar (BUET) |
| Venue | EMNLP 2020, pp. 2612–2623 |
| Why | Both `csebuetnlp/banglat5_nmt_en_bn` and `csebuetnlp/banglat5_nmt_bn_en` are used in your `bangla_t5.py`. The HuggingFace model cards require both citations. |

```bibtex
@inproceedings{Bhattacharjee2023BanglaT5,
  author    = {Bhattacharjee, Abhik and Hasan, Tahmid and
               Ahmad, Wasi Uddin and Shahriyar, Rifat},
  title     = {{BanglaNLG} and {BanglaT5}: Benchmarks and Resources for
               Evaluating Low-Resource Natural Language Generation in Bangla},
  booktitle = {Findings of the Association for Computational Linguistics:
               EACL 2023},
  year      = {2023}
}

@inproceedings{Hasan2020BanglaNMT,
  author    = {Hasan, Tahmid and Bhattacharjee, Abhik and Samin, Kazi and
               Hasan, Masum and Basak, Madhusudan and Rahman, M. Sohel and
               Shahriyar, Rifat},
  title     = {Not Low-Resource Anymore: Aligner Ensembling, Batch Filtering,
               and New Datasets for Bengali-{E}nglish Machine Translation},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in
               Natural Language Processing (EMNLP)},
  pages     = {2612--2623},
  year      = {2020},
  doi       = {10.18653/v1/2020.emnlp-main.207}
}
```

---

## 🟡 Tier 2 — Strongly Recommended (5 papers)

### 11. Cross-Encoder Reranking — FlashRank / ms-marco Lineage
**Nogueira & Cho — arXiv 2019**

| Field | Value |
|---|---|
| Title | Passage Re-ranking with BERT |
| Authors | Nogueira, Cho |
| arXiv | 1901.04085 |
| Why | Your `Reranker` class in `reranker.py` uses `ms-marco-MiniLM-L-12-v2` which is a direct descendant of this cross-encoder reranking approach. |

```bibtex
@misc{Nogueira2019Reranking,
  author        = {Nogueira, Rodrigo and Cho, Kyunghyun},
  title         = {Passage Re-ranking with {BERT}},
  year          = {2019},
  eprint        = {1901.04085},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IR}
}
```

---

### 12. LLM Quantization — Justifies Your Q8 GGUF Choice
**Dettmers et al. — NeurIPS 2022**

| Field | Value |
|---|---|
| Title | LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale |
| Authors | Dettmers, Lewis, Belkada, Zettlemoyer |
| Venue | NeurIPS 2022 |
| Why | You use `q8_0` GGUF quantization. This paper provides the theoretical basis for why 8-bit quantization preserves LLM quality while reducing memory. |

```bibtex
@inproceedings{Dettmers2022LLMint8,
  author    = {Dettmers, Tim and Lewis, Mike and Belkada, Younes and
               Zettlemoyer, Luke},
  title     = {{LLM}.int8(): 8-bit Matrix Multiplication for Transformers
               at Scale},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {35},
  year      = {2022}
}
```

---

### 13. RAG Survey — Framing Reference
**Gao et al. — arXiv 2024**

| Field | Value |
|---|---|
| Title | Retrieval-Augmented Generation for Large Language Models: A Survey |
| Authors | Gao, Xiong, Gao, Jia, Pan, Bi, Dai, Sun, Wang |
| arXiv | 2312.10997 |
| Why | Gives reviewers a comprehensive framing of the RAG landscape your system operates in. Widely cited in 2024–2025 RAG papers. |

```bibtex
@article{Gao2024RAGSurvey,
  author  = {Gao, Yunfan and Xiong, Yun and Gao, Xinyu and Jia, Kangxiang
             and Pan, Jinliu and Bi, Yuxi and Dai, Yi and Sun, Jiawei and
             Wang, Haofen},
  title   = {Retrieval-Augmented Generation for Large Language Models: A Survey},
  journal = {arXiv preprint arXiv:2312.10997},
  year    = {2024}
}
```

---

### 14. Agentic / Iterative RAG — Justifies Your LangGraph Self-Correction Loop
**Trivedi et al. — ACL 2023**

| Field | Value |
|---|---|
| Title | Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions |
| Authors | Trivedi, Balasubramanian, Khot, Sabharwal |
| Venue | ACL 2023 |
| arXiv | 2212.10509 |
| Why | Directly motivates your LangGraph `retrieve → grade → rewrite → retrieve` bounded retry loop. Provides academic backing for your agentic design choice. |

```bibtex
@inproceedings{Trivedi2023IRCoT,
  author    = {Trivedi, Harsh and Balasubramanian, Niranjan and
               Khot, Tushar and Sabharwal, Ashish},
  title     = {Interleaving Retrieval with Chain-of-Thought Reasoning for
               Knowledge-Intensive Multi-Step Questions},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for
               Computational Linguistics (ACL)},
  year      = {2023},
  eprint    = {2212.10509},
  archivePrefix = {arXiv}
}
```

---

### 15. BanglaBERT — Bengali NLP Context
**Bhattacharjee et al. — NAACL 2022**

| Field | Value |
|---|---|
| Title | BanglaBERT: Language Model Pretraining and Benchmarks for Low-Resource Language Understanding Evaluation in Bangla |
| Authors | Bhattacharjee, Hasan, Ahmad, Mubasshir, Islam, Iqbal, Rahman, Shahriyar (BUET) |
| Venue | Findings of NAACL 2022 |
| Why | Establishes Bengali as a low-resource language requiring specialized NLP treatment. Directly motivates your KG dialect-bridging contribution and BanglaT5 translation choice. |

```bibtex
@inproceedings{Bhattacharjee2022BanglaBERT,
  author    = {Bhattacharjee, Abhik and Hasan, Tahmid and Ahmad, Wasi Uddin
               and Mubasshir, Kazi and Islam, Md. Saiful and Iqbal, Anindya
               and Rahman, M. Sohel and Shahriyar, Rifat},
  title     = {{BanglaBERT}: Language Model Pretraining and Benchmarks for
               Low-Resource Language Understanding Evaluation in Bangla},
  booktitle = {Findings of the Annual Conference of the North American Chapter
               of the Association for Computational Linguistics (NAACL 2022)},
  year      = {2022}
}
```

---

## 🟢 Tier 3 — Recommended for Completeness (3 papers)

### 16. ONNX Runtime — Your Vision Inference Engine
**ONNX Runtime developers, 2023**

| Field | Value |
|---|---|
| Why | Your `classifier.py` uses `onnxruntime` with `CPUExecutionProvider` for MobileNetV3+CBAM inference. Software citation. |

```bibtex
@misc{ONNXRuntime2023,
  author = {{ONNX Runtime developers}},
  title  = {{ONNX Runtime}},
  year   = {2023},
  url    = {https://onnxruntime.ai},
  note   = {Version 1.x}
}
```

---

### 17. Agricultural ICT in Developing Countries — Domain Motivation
**Wolfert et al. — Agricultural Systems 2017**

| Field | Value |
|---|---|
| Title | Big Data in Smart Farming — A Review |
| Authors | Wolfert, Ge, Verdouw, Bogaardt |
| Venue | Agricultural Systems, Vol. 153, pp. 69–80 |
| DOI | 10.1016/j.agsy.2017.01.023 |
| Why | Motivates why AI-driven agricultural advisory systems matter globally. Sets domain context in your Introduction section. |

```bibtex
@article{Wolfert2017BigDataAgriculture,
  author    = {Wolfert, Sjaak and Ge, Lan and Verdouw, Cor and
               Bogaardt, Marc-Jeroen},
  title     = {Big Data in Smart Farming -- A Review},
  journal   = {Agricultural Systems},
  volume    = {153},
  pages     = {69--80},
  year      = {2017},
  doi       = {10.1016/j.agsy.2017.01.023}
}
```

---

### 18. Knowledge Graphs — General Reference
**Hogan et al. — ACM Computing Surveys 2021**

| Field | Value |
|---|---|
| Title | Knowledge Graphs |
| Authors | Hogan, Blomqvist, Cochez, et al. |
| Venue | ACM Computing Surveys, Vol. 54, No. 4, pp. 1–37 |
| DOI | 10.1145/3447772 |
| Why | Provides formal grounding for your SQLite-backed Knowledge Graph design (entities, aliases, relations). Useful when describing your KG schema in Related Work. |

```bibtex
@article{Hogan2021KnowledgeGraphs,
  author    = {Hogan, Aidan and Blomqvist, Eva and Cochez, Michael and
               d'Amato, Claudia and Melo, Gerard de and Gutierrez, Claudio
               and Kirrane, Sabrina and Gayo, Jos{\'e} Emilio Labra and
               Navigli, Roberto and Neumaier, Sebastian and
               Ngomo, Axel-Cyrille Ngonga and Polleres, Axel and
               Rashid, Sabbir M. and Rula, Anisa and Schmelzeisen, Lukas and
               Sequeda, Juan and Staab, Steffen and Zimmermann, Antoine},
  title     = {Knowledge Graphs},
  journal   = {ACM Computing Surveys},
  volume    = {54},
  number    = {4},
  pages     = {1--37},
  year      = {2021},
  doi       = {10.1145/3447772}
}
```

---

## Summary Table

| # | Paper | Authors | Venue | Tier | Status |
|---|---|---|---|---|---|
| 1 | RAG (original) | Lewis et al. | NeurIPS 2020 | 🔴 | Missing |
| 2 | FAISS | Johnson et al. | IEEE Trans. Big Data 2021 | 🔴 | Missing |
| 3 | BM25 | Robertson & Zaragoza | FnTIR 2009 | 🔴 | Missing |
| 4 | RRF | Cormack et al. | SIGIR 2009 | 🔴 | Missing |
| 5 | Sentence-BERT | Reimers & Gurevych | EMNLP 2019 | 🔴 | Missing |
| 6 | Whisper | Radford et al. | ICML 2023 | 🔴 | Missing |
| 7 | MobileNetV3 | Howard et al. | ICCV 2019 | 🔴 | Missing |
| 8 | CBAM | Woo et al. | ECCV 2018 | 🔴 | Missing |
| 9 | Qwen2.5 | Qwen Team | arXiv 2025 | 🔴 | Missing |
| 10a | BanglaT5 | Bhattacharjee et al. | EACL 2023 | 🔴 | Missing |
| 10b | BanglaNMT corpus | Hasan et al. | EMNLP 2020 | 🔴 | Missing |
| 11 | Cross-encoder rerank | Nogueira & Cho | arXiv 2019 | 🟡 | Missing |
| 12 | LLM.int8() quantization | Dettmers et al. | NeurIPS 2022 | 🟡 | Missing |
| 13 | RAG Survey | Gao et al. | arXiv 2024 | 🟡 | Missing |
| 14 | Agentic RAG (IRCoT) | Trivedi et al. | ACL 2023 | 🟡 | Missing |
| 15 | BanglaBERT | Bhattacharjee et al. | NAACL 2022 | 🟡 | Missing |
| 16 | ONNX Runtime | ONNX developers | Software 2023 | 🟢 | Missing |
| 17 | Big Data in Smart Farming | Wolfert et al. | Agri. Systems 2017 | 🟢 | Missing |
| 18 | Knowledge Graphs | Hogan et al. | ACM CSUR 2021 | 🟢 | Missing |

---

## Final Count

```
Already reviewed:   16 papers  ✅
Tier 1 to add:      11 papers  🔴  (non-negotiable)
Tier 2 to add:       5 papers  🟡  (strongly recommended)
Tier 3 to add:       3 papers  🟢  (optional but helpful)
─────────────────────────────────
Total after adding: ~35 papers  ✅  IEEE conference ready
```

> **Note:** Verify BanglaT5 (items 10a, 10b) and BanglaBERT (item 15) exact page numbers from ACL Anthology before submitting. All other BibTeX entries above are confirmed from official sources.