import subprocess
subprocess.check_call(["pip", "install", "pypdf"])
from pypdf import PdfReader
reader = PdfReader("Offline Multimodal Agentic RAG for Low-Resource Bilingual Agricultural Decision Support.pdf")
text = [page.extract_text() for page in reader.pages]
with open("report_text.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(text))
