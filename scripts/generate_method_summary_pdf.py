import re
import os
import subprocess
from markdown_it import MarkdownIt

with open("docs/METHOD_SUMMARY_2PAGE.md", "r") as f:
    text = f.read()

# Protect math blocks
math_store = []
def repl_disp(m):
    idx = len(math_store)
    math_store.append(m.group(0))
    return f"%%MATH_DISP_{idx}%%"

def repl_inline(m):
    idx = len(math_store)
    math_store.append(m.group(0))
    return f"%%MATH_INL_{idx}%%"

text = re.sub(r"\$\$(.*?)\$\$", repl_disp, text, flags=re.DOTALL)
text = re.sub(r"\$(.*?)\$", repl_inline, text)

# Remove the raw markdown title and meta since we'll make a sleek HTML header
# Lines 1 to 5
lines = text.splitlines()
body_lines = [l for l in lines if not l.startswith("# Method Summary") and not l.startswith("**Team**:") and not l.startswith("**Leaderboard") and not l.strip() == "---"]
clean_text = "\n".join(body_lines).strip()

md = MarkdownIt("gfm-like", {"linkify": False, "breaks": True})
html_body = md.render(clean_text)

# Restore math
for i, m in enumerate(math_store):
    html_body = html_body.replace(f"%%MATH_DISP_{i}%%", m)
    html_body = html_body.replace(f"%%MATH_INL_{i}%%", m)

# Put clean page break right before Section 3
html_body = html_body.replace(
    "<h3>3. Model Architecture",
    "<div class=\"page-break\"></div>\n<h3>3. Model Architecture"
)

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Method Summary: Multi-Horizon Tree-Neural Hybrid with Temporal Entity Propagation</title>
<!-- KaTeX -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
<script>
    document.addEventListener("DOMContentLoaded", function() {{
        renderMathInElement(document.body, {{
            delimiters: [
                {{left: '$$', right: '$$', display: true}},
                {{left: '$', right: '$', display: false}}
            ],
            throwOnError: false
        }});
        window.status = "ready";
    }});
</script>
<style>
    @page {{
        size: A4;
        margin: 13mm 16mm 13mm 16mm;
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 9.15pt;
        line-height: 1.38;
        color: #1e293b;
        margin: 0;
        padding: 0;
    }}
    .page-break {{
        page-break-before: always;
        break-before: page;
    }}
    .header-card {{
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        background: #f8fafc;
        padding: 10px 14px;
        margin-bottom: 10px;
    }}
    .header-title {{
        font-size: 14.5pt;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 6px 0;
        line-height: 1.25;
        letter-spacing: -0.2px;
    }}
    .meta-grid {{
        display: grid;
        grid-template-columns: 1.1fr 1fr;
        gap: 4px 12px;
        font-size: 8.8pt;
        color: #475569;
    }}
    .meta-grid strong {{
        color: #0f172a;
    }}
    .badge {{
        display: inline-block;
        background: #e2e8f0;
        color: #0f172a;
        font-weight: 600;
        padding: 0px 5px;
        border-radius: 3px;
    }}
    h3 {{
        font-size: 10.4pt;
        margin: 7px 0 2.5px 0;
        color: #0f172a;
        font-weight: 700;
        border-bottom: 1.5px solid #cbd5e1;
        padding-bottom: 2px;
        page-break-after: avoid;
    }}
    p {{
        margin: 2.5px 0 3.5px 0;
    }}
    ul, ol {{
        margin: 2.5px 0 3.5px 0;
        padding-left: 17px;
    }}
    li {{
        margin-bottom: 2px;
    }}
    code {{
        background-color: #f1f5f9;
        color: #0f172a;
        padding: 1px 3px;
        border-radius: 3px;
        font-family: "JetBrains Mono", Consolas, Menlo, monospace;
        font-size: 8.3pt;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 6px 0;
        font-size: 8.5pt;
        page-break-inside: avoid;
    }}
    th, td {{
        border: 1px solid #cbd5e1;
        padding: 3.5px 6px;
        text-align: left;
    }}
    th {{
        background-color: #f1f5f9;
        font-weight: 600;
        color: #1e293b;
    }}
    tr:nth-child(even) {{
        background-color: #f8fafc;
    }}
    .katex-display {{
        margin: 4px 0 !important;
    }}
</style>
</head>
<body>

<div class="header-card">
    <div class="header-title">Method Summary: Multi-Horizon Tree-Neural Hybrid with Temporal Entity Propagation</div>
    <div class="meta-grid">
        <div><strong>Team:</strong> Overfit & Overcaffeinated</div>
        <div><strong>Competition:</strong> REACT 2026 Datathon (Tabular Fraud Detection)</div>
        <div><strong>Public Leaderboard:</strong> <span class="badge">0.56548</span></div>
        <div><strong>Local PR-AUC (Jul 1–15):</strong> <span class="badge">0.5359</span> (Baseline: 0.1664)</div>
    </div>
</div>

{html_body}
</body>
</html>
"""

html_path = os.path.abspath("docs/METHOD_SUMMARY.html")
pdf_path = os.path.abspath("docs/METHOD_SUMMARY.pdf")

with open(html_path, "w") as f:
    f.write(full_html)

subprocess.run([
    "chromium",
    "--headless",
    "--disable-gpu",
    "--no-sandbox",
    "--run-all-compositor-stages-before-draw",
    "--virtual-time-budget=3000",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_path}",
    f"file://{html_path}"
], check=True)

subprocess.run(["pdfinfo", pdf_path], check=True)
subprocess.run([
    "pdftoppm", "-png", "-r", "150", pdf_path,
    "/home/sanzid/.gemini/antigravity-cli/brain/50fbf02e-94b0-4708-92f1-540296ba4f03/scratch/page_v6"
], check=True)
