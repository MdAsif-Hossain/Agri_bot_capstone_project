"""Fig. 2 generator --- LangGraph agentic workflow (Colab snippet).

Reproduces the paper's Fig. 2. The `translate` node renders EN->BN with a
domain-adapted BanglaT5; BN->EN input translation is stock BanglaT5 and
happens in `normalize`.

Run in Colab (one cell):
    !apt-get -qq install -y graphviz
    !pip -q install graphviz
    # then paste the body below and run
Writes fig2.png + fig2.pdf and (in Colab) downloads them.
"""

import graphviz


def generate_workflow():
    g = graphviz.Digraph("workflow", format="png", engine="dot")
    g.attr(rankdir="TB", bgcolor="white", fontname="Helvetica", fontsize="11",
           nodesep="0.35", ranksep="0.45")
    g.attr("node", shape="box", style="filled", fontname="Helvetica",
           fontsize="11", color="#888888", penwidth="1.0")
    g.attr("edge", fontname="Helvetica", fontsize="9", color="#555555",
           arrowsize="0.7")

    BLUE, YELLOW, PINK = "#E8EEF7", "#FFF2CC", "#F8D7DA"
    ORANGE, GREEN, GRAY = "#FCE5CD", "#E0F0E0", "#CCCCCC"

    g.node("start", "Input", shape="ellipse", fillcolor=GRAY)
    g.node("normalize", "normalize  (BN → EN)", fillcolor=BLUE)
    g.node("kg_link", "kg_link  (entity expand)", fillcolor=BLUE)
    g.node("retrieve", "retrieve  (FAISS + BM25 + RRF)", fillcolor=BLUE)
    g.node("rerank", "rerank  (cross-encoder)", fillcolor=BLUE)
    g.node("grade", "grade  (overlap fast-path + LLM)",
           shape="diamond", fillcolor=YELLOW)
    g.node("rewrite", "rewrite\n(LLM)", fillcolor=PINK)
    g.node("generate", "generate  (Qwen2.5-3B, GPU)", fillcolor=ORANGE)
    g.node("translate", "translate  (BanglaT5)", fillcolor=BLUE)
    g.node("verify", "verify  (LLM + grounding backstop)", fillcolor=GREEN)
    g.node("enforce", "enforce_policy  (two-factor dosage refusal)",
           fillcolor=GREEN)
    g.node("end", "Output", shape="ellipse", fillcolor=GRAY)

    g.edge("start", "normalize")
    g.edge("normalize", "kg_link")
    g.edge("kg_link", "retrieve")
    g.edge("retrieve", "rerank")
    g.edge("rerank", "grade")
    g.edge("grade", "rewrite", label="INSUFF.\n(retry<2)")
    g.edge("rewrite", "retrieve", label="re-query", style="dashed",
           color="#B04A4A", fontcolor="#B04A4A", constraint="false")
    g.edge("grade", "generate", label="SUFFICIENT / max retries")
    g.edge("generate", "translate")
    g.edge("translate", "verify")
    g.edge("verify", "enforce")
    g.edge("enforce", "end")

    g.render("fig2", format="png", cleanup=True)
    g.render("fig2", format="pdf", cleanup=True)
    print("wrote fig2.png / fig2.pdf")
    return "fig2.png"


if __name__ == "__main__":
    output_file = generate_workflow()
    # Auto-download in Colab; ignored when run locally.
    try:
        from google.colab import files
        for f in ("fig2.png", "fig2.pdf"):
            files.download(f)
    except Exception:
        print(f"Generated: {output_file}")
