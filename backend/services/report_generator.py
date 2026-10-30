import csv
import io
import json

def flatten(result):
    return [dict(document_a=r["a"]["filename"],page_a=r["a"]["page"] or "N/A",
        statement_a=r["a"]["text"],context_a=r["a"]["parent"],
        document_b=r["b"]["filename"],page_b=r["b"]["page"] or "N/A",
        statement_b=r["b"]["text"],context_b=r["b"]["parent"],classification=r["classification"],
        **r["probabilities"],explanation=r["explanation"]) for r in result["results"]]

def generate(result,format):
    if format=="json":
        return json.dumps(result,indent=2,ensure_ascii=False).encode(),"application/json"
    if format=="csv":
        stream=io.StringIO(newline="")
        rows=flatten(result)
        fields=list(rows[0]) if rows else ["document_a","statement_a","document_b","statement_b","classification","contradiction","explanation"]
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        # Prevent spreadsheet formulas originating in untrusted documents.
        for row in rows:
            writer.writerow({k:("'"+v if isinstance(v,str) and v.lstrip().startswith(("=","+","-","@")) else v) for k,v in row.items()})
        return stream.getvalue().encode("utf-8-sig"),"text/csv"
    if format=="pdf":
        import pymupdf
        from html import escape
        content="<h1>DocuConflict AI report</h1>"
        names=result.get("document_names") or sorted({r[side]["filename"] for r in result["results"] for side in ("a","b")})
        for key,value in {"Documents":", ".join(names),"Timestamp":result["timestamp"],
            "Model":result["model"],"Embedding model":result["embedding_model"],
            "Candidate pairs":result["candidate_pairs"],"Threshold":result["threshold"],
            "Top-K":result["top_k"]}.items():
            content+=f"<p><b>{escape(key)}:</b> {escape(str(value))}</p>"
        content+="<p>Scores are uncalibrated probabilities. Review source context.</p>"
        for row in flatten(result):
            content+=f'<h2>{escape(row["classification"])} ({row["contradiction"]:.3f} contradiction)</h2>'
            for side in ("a","b"):
                content+=f'<p><b>{escape(row["document_"+side])} ? page {row["page_"+side]}</b><br>{escape(row["statement_"+side])}</p>'
                content+=f'<p>Context: {escape(row["context_"+side])}</p>'
            content+=f'<p>{escape(row["explanation"])}</p>'
        stream=io.BytesIO()
        writer=pymupdf.DocumentWriter(stream)
        story=pymupdf.Story(html=content)
        media=pymupdf.paper_rect("a4")
        rect=media+(36,36,-36,-36)
        more=1
        while more:
            device=writer.begin_page(media)
            more,_=story.place(rect)
            story.draw(device)
            writer.end_page()
        writer.close()
        return stream.getvalue(),"application/pdf"
    raise ValueError("Report format must be csv, json, or pdf.")
