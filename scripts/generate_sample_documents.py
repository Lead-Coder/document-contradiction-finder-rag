import json
from backend.core.config import DATA

# All text below is synthetic, manually authored evaluation material.
CASES=[
("upload_limit","The maximum file upload size is 20 MB.","The maximum file upload size is 10 MB.",True,"numerical"),
("late_work","Students are allowed to submit assignments after the deadline.","Students are not allowed to submit assignments after the deadline.",True,"negation"),
("leave","Employees receive 20 days of annual leave.","The annual leave entitlement for employees is 20 days.",False,"agreement"),
("fee_period","The registration fee was INR 500 in 2024.","The registration fee is INR 700 in 2026.",False,"different_years"),
("unrelated","The library opens at 9 AM.","The cafeteria serves breakfast from 8 AM.",False,"unrelated"),
("remote","Remote work is permitted on Fridays.","Employees must work on-site every Friday.",True,"paraphrase"),
("budgets","Department A has a budget of INR 50,000.","Department B has a budget of INR 70,000.",False,"different_entities"),
("refund","Customers can request a refund within 30 days of purchase.","Customers cannot request a refund within 30 days of purchase.",True,"refund_policy"),
("deadline","Project Atlas must be delivered by June 15, 2026.","Project Atlas must be delivered by July 15, 2026.",False,"compatible_deadline_upper_bounds"),
("requirements","The server must run Python 3.11 or later.","The server must run Python 3.12 or later.",False,"more_specific_requirement"),
("conditions","Students receive a fee waiver if their family income is low.","Students do not receive a fee waiver if their family income is high.",False,"different_conditions"),
("missing","The library offers wireless internet.","The library has a reading room.",False,"missing_information"),
("exam","The final examination starts at 10 AM on May 20, 2026.","The final examination starts at 2 PM on May 20, 2026.",True,"academic_schedule"),
("storage","Product Nova includes exactly 256 GB of storage.","Product Nova includes exactly 512 GB of storage.",True,"product_specification"),
("policy","Employees must wear identification badges inside the office.","Employees are not required to wear identification badges inside the office.",True,"company_policy"),
("dates","The enrollment office is closed on January 1, 2026.","The enrollment office is open on January 2, 2026.",False,"different_dates"),
]

def main():
    root=DATA/"benchmarks";root.mkdir(exist_ok=True,parents=True)
    manifest=[]
    for name,left,right,conflict,category in CASES:
        documents=[]
        for side,text in (("a",left),("b",right)):
            filename=f"{name}_{side}.txt"
            (root/filename).write_text(text,encoding="utf-8")
            documents.append(dict(filename=filename,text=text))
        manifest.append(dict(id=name,category=category,documents=documents,
            expected_conflicts=[[left,right]] if conflict else [],synthetic=True))
    multi_left=["The campus has a landscaped garden.","The maximum file upload size is 20 MB.","The reading room contains reference books.","Remote work is permitted on Fridays."]
    multi_right=["The reception desk provides visitor information.","The maximum file upload size is 10 MB.","The campus has a bicycle parking area.","Employees must work on-site every Friday."]
    docs=[]
    import pymupdf
    for side,paragraphs in (("a",multi_left),("b",multi_right)):
        filename=f"multipage_{side}.pdf"
        doc=pymupdf.open()
        for text in paragraphs:
            page=doc.new_page();page.insert_textbox(pymupdf.Rect(50,50,550,750),text,fontsize=12)
        doc.save(root/filename);doc.close()
        docs.append(dict(filename=filename,text="\n\n".join(paragraphs)))
    manifest.append(dict(id="multipage",category="multi_page",documents=docs,synthetic=True,
        expected_conflicts=[[multi_left[1],multi_right[1]],[multi_left[3],multi_right[3]]]))
    third=["The maximum file upload size is 20 MB.","The maximum file upload size is 10 MB.","The maximum file upload size is 20 MB."]
    docs=[]
    for i,text in enumerate(third):
        filename=f"three_documents_{i}.txt"
        (root/filename).write_text(text,encoding="utf-8")
        docs.append(dict(filename=filename,text=text))
    manifest.append(dict(id="three_documents",category="three_documents",documents=docs,synthetic=True,
        expected_conflicts=[[third[0],third[1]]]))
    (root/"ground_truth.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    samples=DATA/"samples";samples.mkdir(exist_ok=True)
    from docx import Document
    for side,texts in (("a",multi_left),("b",multi_right)):
        doc=Document()
        for text in texts:doc.add_paragraph(text)
        doc.save(samples/f"demo_policy_{side}.docx")
        (samples/f"demo_policy_{side}.txt").write_text("\n\n".join(texts),encoding="utf-8")
    print(f"Generated {len(manifest)} synthetic benchmark cases, including multi-page PDFs and three-document comparisons.")

if __name__=="__main__":main()
