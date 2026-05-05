from pypdf import PdfReader
r = PdfReader("Helene PMP HUB.pdf")
for i, p in enumerate(r.pages):
    print(f"--- PAGE {i+1} ---")
    print(p.extract_text())
