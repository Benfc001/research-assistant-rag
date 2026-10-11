import pandas as pd
from generate_answer import retrieve, llm, MODEL

N_QUESTIONS = 20
TOP_K = 5

# ---------- Load papers and pick a random sample ----------
df = pd.read_csv("arxiv_clean.csv")
needed = {"arxiv_id", "title", "abstract"}
missing = needed - set(df.columns)
if missing:
    raise SystemExit(f"Missing column(s) {missing}. Columns in file: {list(df.columns)}")

sample = df.sample(n=N_QUESTIONS, random_state=42)  # random_state = same sample every run


# ---------- Have Claude write a realistic test question from an abstract ----------
def make_question(abstract):
    prompt = f"""Read this research paper abstract. Write ONE short, general question (under 15 words)
that someone who has NOT read this paper might ask, where this paper would be a good answer.
Use everyday wording. Do not use the paper's method names, dataset names, or technical phrases.
Return only the question.

Abstract:
{abstract}"""
    response = llm.messages.create(
        model=MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


# ---------- Run the evaluation ----------
results = []
for n, row in enumerate(sample.itertuples(), 1):
    question = make_question(row.abstract)
    papers = retrieve(question, top_k=TOP_K)
    retrieved_ids = [p["arxiv_id"] for p in papers]

    rank = retrieved_ids.index(row.arxiv_id) + 1 if row.arxiv_id in retrieved_ids else None
    results.append({
        "arxiv_id": row.arxiv_id,
        "title": row.title,
        "question": question,
        "rank": rank,
    })
    status = f"found at rank {rank}" if rank else "NOT found"
    print(f"[{n}/{N_QUESTIONS}] {status} | {question[:80]}")

# ---------- Score it ----------
res = pd.DataFrame(results)
hit_rate = res["rank"].notna().mean()
mrr = res["rank"].apply(lambda r: 1 / r if pd.notna(r) else 0).mean()

print("\n========== RESULTS (HARD MODE) ==========")
print(f"Questions tested : {N_QUESTIONS}")
print(f"Hit rate @{TOP_K}     : {hit_rate:.0%}")
print(f"MRR              : {mrr:.2f}")

res.to_csv("eval_results_hard.csv", index=False)
print("Details saved to eval_results_hard.csv")