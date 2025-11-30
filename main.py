import os, re, json
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.api.types import EmbeddingFunction

# ----------------- CONFIG -----------------
FILE_METADATA = {
    "ADA_Behaviors.txt": {"domain": "behaviors", "population": "adults"},
    "ADA_OlderAdults.txt": {"domain": "older_adults", "population": "older_adults"},
    "ADA_Obesity.txt": {"domain": "obesity", "population": "adults"},
}

VARIABLE_TO_DOMAIN = {
    "MVPA": "behaviors",
    "Sugary_drinks_per_day": "behaviors",
    "BMI": "obesity",
    "Waist": "obesity",
    "Triglycerides": "obesity",
    "HbA1c": "behaviors",
}

# ----------------- HELPERS -----------------
def chunk_text(text: str, size: int = 900, overlap: int = 120) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += (size - overlap)
    return chunks


class OpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model

    def __call__(self, input: List[str]):
        if isinstance(input, str):
            texts = [input]
        else:
            texts = input
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


def get_or_build_vectorstore():
    db_path = "chroma_db"
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    emb = OpenAIEmbeddingFunction()

    existing = client.list_collections()
    if "ada" in existing:
        print("Found existing 'ada' collection. Reusing vectorstore.")
        return client.get_collection("ada", embedding_function=emb)

    print("No existing 'ada' collection. Building vectorstore from scratch...")
    col = client.create_collection(name="ada", embedding_function=emb)

    for fname in os.listdir("resources"):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join("resources", fname), "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text)
        meta = FILE_METADATA.get(fname, {})
        for i, ch in enumerate(chunks):
            col.add(ids=[f"{fname}_{i}"], documents=[ch], metadatas=[meta])
        print(f"Loaded {fname}: {len(chunks)} chunks")
    print("Vectorstore build complete.")
    return col


def parse_patient_file(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    parts = re.split(r"(CLINICAL INFO|RL ACTION PLAN|PATIENT CONTEXT)", text)
    out = {"clinical": "", "rl": "", "context": ""}
    for i in range(len(parts)):
        if parts[i] == "CLINICAL INFO":
            out["clinical"] = parts[i + 1].strip()
        elif parts[i] == "RL ACTION PLAN":
            out["rl"] = parts[i + 1].strip()
        elif parts[i] == "PATIENT CONTEXT":
            out["context"] = parts[i + 1].strip()
    return out


def extract_rl_actions(rl_text: str) -> List[Dict]:
    steps = rl_text.split("Step")
    actions = []
    for s in steps:
        if "Variable:" not in s:
            continue
        var = re.search(r"Variable:\s*(.+)", s)
        baseline = re.search(r"Baseline value:\s*(.+)", s)
        delta = re.search(r"Suggested delta:\s*(.+)", s)
        target = re.search(r"Target value:\s*(.+)", s)
        actions.append(
            {
                "variable": var.group(1).strip() if var else "",
                "baseline": baseline.group(1).strip() if baseline else "",
                "delta": delta.group(1).strip() if delta else "",
                "target": target.group(1).strip() if target else "",
            }
        )
    return actions


def retrieve_guidelines(collection, domain: str, population: str, query: str) -> str:
    conds = []
    if domain:
        conds.append({"domain": {"$eq": domain}})
    if population:
        conds.append({"population": {"$eq": population}})

    if conds:
        where = {"$and": conds}
        res = collection.query(query_texts=[query], n_results=4, where=where)
    else:
        res = collection.query(query_texts=[query], n_results=4)

    docs = res.get("documents", [[]])[0] or []
    return "\n\n".join(docs)


def llm_generate(
    clinical: str,
    context: str,
    rl_actions: List[Dict],
    evidence_blocks: List[Dict],
) -> str:
    client = OpenAI()

    system_prompt = (
        "You are a warm, supportive diabetes lifestyle coach who ALWAYS stays inside ADA guidelines. "
        "You NEVER recommend medication changes or adjust prescriptions. "
        "You speak directly to the patient in clear, everyday language, roughly at a 7th–8th grade reading level. "
        "Use short sentences. Avoid terms like 'glycemic control', 'cardiometabolic', or 'dyslipidemia'. "
        "Instead, say things like 'blood sugar', 'heart and blood vessels', or 'cholesterol and fats in the blood'. "
        "You acknowledge their real-life constraints (work schedule, money, mobility, stress) and say out loud that "
        "some changes may be hard, then suggest realistic options anyway.\n\n"
        "For EACH RL action, you must:\n"
        "- Address the patient as 'you'.\n"
        "- Be encouraging, non-judgmental, and practical.\n"
        "- Be specific about behaviors (e.g., 'brisk walking for 10 minutes', '2 sets of 10 sit-to-stands').\n"
        "- Explicitly connect your plan to the patient’s context.\n"
        "- Stay consistent with ADA Standards of Care for lifestyle and behavior."
    )

    user_prompt = {
        "clinical_info": clinical,
        "patient_context": context,
        "rl_actions": rl_actions,
        "evidence": evidence_blocks,
        "task": (
            "Write a numbered list, one item per RL action, speaking DIRECTLY to the patient.\n\n"
            "For each item, use this structure with VERY clear labels and short paragraphs:\n\n"
            "1) A short heading like: '1. Moving more: from 15 to 60 minutes per week'.\n\n"
            "Then four labeled parts:\n"
            "- 'What this step means:' One short explanation of what you are asking the patient to do.\n"
            "- 'Why this helps your health:' 1–2 simple sentences about blood sugar, heart, weight, or future problems.\n"
            "- 'A realistic plan for you:' 3–4 sentences that:\n"
            "   * Start by acknowledging that this may be hard in their situation (for example, night shifts, little money, pain, housing), and\n"
            "   * Give 2–3 concrete actions with WHEN and HOW (for example, 'After your shift, walk 10 minutes from the bus stop', "
            "     'On days off, do 2 sets of 10 sit-to-stands from a chair while watching a video', "
            "     'During your break, swap one energy drink for water').\n"
            "- 'How this fits ADA guidelines:' 1 short sentence saying this matches ADA diabetes lifestyle guidance, in plain language "
            "(for example, 'This follows ADA diabetes guidelines that encourage more movement and less sitting to protect your heart and blood sugar.').\n\n"
            "Keep each item under about 120–150 words. Avoid technical jargon. Do NOT talk to a clinician; talk directly to the patient."
        ),
    }

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt)},
        ],
        temperature=0.2,
        max_tokens=950,
    )
    return resp.choices[0].message.content


# ----------------- MAIN -----------------
def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found. Put it in a .env file.")

    print("\n[1] Loading or building vectorstore from ADA guideline text files...")
    collection = get_or_build_vectorstore()

    print("\n[2] Loading patient snapshot from input/sample2.txt...")
    data = parse_patient_file("input/sample2.txt")
    clinical, rl_text, context = data["clinical"], data["rl"], data["context"]

    print("\n[3] Parsing RL action plan...")
    rl_actions = extract_rl_actions(rl_text)

    age_match = re.search(r"Age.*?:\s*(\d+)", clinical)
    pop = "older_adults" if age_match and int(age_match.group(1)) >= 65 else "adults"
    print(f"Detected population subgroup: {pop}")

    print("\n[4] Retrieving ADA evidence for each RL step...")
    evidence_blocks: List[Dict] = []
    for step in rl_actions:
        var = step["variable"]
        domain = None
        for key, dom in VARIABLE_TO_DOMAIN.items():
            if key.lower() in var.lower():
                domain = dom
                break
        if not domain:
            domain = "behaviors"
        q = f"ADA guideline for {var}, lifestyle change, diabetes, feasibility"
        ev = retrieve_guidelines(collection, domain, pop, q)
        evidence_blocks.append(
            {"variable": var, "domain": domain, "population": pop, "evidence": ev}
        )

    print("\n[5] Calling LLM to generate final recommendations...")
    final_output = llm_generate(clinical, context, rl_actions, evidence_blocks)

    print("\n==============================")
    print("FINAL RECOMMENDATION")
    print("==============================\n")
    print(final_output)

    print("\n==============================")
    print("EVIDENCE USED (RAW ADA CHUNKS)")
    print("==============================\n")
    for i, ev in enumerate(evidence_blocks, start=1):
        print(f"--- RL Action {i}: {ev['variable']} ---")
        print(ev["evidence"] or "(No evidence chunks retrieved)")
        print()

    print("Done.\n")


if __name__ == "__main__":
    main()