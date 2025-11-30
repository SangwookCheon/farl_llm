import os
import re
import json
from typing import List, Dict, Any

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

# Map variable names (lowercase) to guideline domains
VARIABLE_TO_DOMAIN = {
    "mvpa": "behaviors",
    "sugary_drinks_per_day": "behaviors",
    "bmi": "obesity",
    "waist": "obesity",      # generic
    "wst": "obesity",        # waist in your dict
    "triglycerides": "obesity",
    "tyg": "obesity",        # if treating as lipid-related
    "hba1c": "behaviors",
    "hb1ac": "behaviors",    # spelling in your dict
}

# ----------------- PATIENT OUTPUT DICT (EXAMPLE) -----------------
result = {
    'gender':1.0,'age':56.0,'race':2.0,'educ':3.0,'marry':1.0,
    'house':1.0,'pov':2.15,'wt':96.4,'ht':168.0,'bmi':34.1,
    'wst':110.5,'hip':112.0,'dia':84.0,'pulse':78.0,'sys':138.0,
    'alt':42.0,'albumin':4.3,'ast':35.0,'crea':0.98,'chol':205.0,
    'tyg':230.0,'ggt':48.0,'wbc':7.2,'hb':14.2,'hct':43.0,
    'ldl':132.0,'hdl':38.0,'acratio':3.1,'glu':182.0,
    'insulin':18.5,'crp':4.8,'hb1ac':8.6,'mvpa':40.0,
    'ac_week':0.0,
    'context':'Middle-aged patient with long-standing type 2 diabetes, obesity, inconsistent follow-up, and limited financial resources. Frequently eats late meals due to shift-based work schedule, with high intake of refined carbohydrates and sugary beverages. Reports chronic knee discomfort limiting high-impact exercise. Demonstrates motivation to prevent future complications but experiences difficulty implementing recommendations without clear, structured steps.',
    "1week":("mvpa",20.0),
    "2week":("ac_week",1.0),
    "3week":("sugary_drinks_per_day",-0.5),
    "4week":("wst",-1.0),
    "5week":("glu",-5.0),
    "6week":("hb1ac",-0.1),
    "7week":("tyg",-10.0),
    "8week":("sys",-2.0),
    "old_score":103.0,
    "new_score":93.0
}

# ----------------- HELPERS -----------------
def chunk_text_by_headings(
    text: str,
    max_chars: int = 900,
    overlap: int = 120,
) -> List[str]:
    lines = text.splitlines()
    heading_pattern = re.compile(r"^(#{1,4})\s+.+")  # #, ##, ###, #### + space + text

    sections: List[str] = []
    current_section: List[str] = []

    for line in lines:
        if heading_pattern.match(line):
            if current_section:
                section_text = "\n".join(current_section).strip()
                if section_text:
                    sections.append(section_text)
                current_section = []
            current_section.append(line)
        else:
            current_section.append(line)

    if current_section:
        section_text = "\n".join(current_section).strip()
        if section_text:
            sections.append(section_text)

    chunks: List[str] = []
    for sec in sections:
        if len(sec) <= max_chars:
            chunks.append(sec)
        else:
            start = 0
            while start < len(sec):
                end = start + max_chars
                chunk = sec[start:end]
                chunks.append(chunk)
                start += max_chars - overlap

    return chunks


def chunk_text(text: str, size: int = 900, overlap: int = 120) -> List[str]:
    return chunk_text_by_headings(text, max_chars=size, overlap=overlap)


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

    try:
        col = client.get_collection("ada", embedding_function=emb)
        print("Found existing 'ada' collection. Reusing vectorstore.")
        return col
    except Exception:
        print("No existing 'ada' collection. Building vectorstore from scratch...")

    col = client.create_collection(name="ada", embedding_function=emb)

    for fname in os.listdir("resources"):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join("resources", fname), "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text, size=900, overlap=120)
        meta = FILE_METADATA.get(fname, {})

        for i, ch in enumerate(chunks):
            col.add(ids=[f"{fname}_{i}"], documents=[ch], metadatas=[meta])

        print(f"Loaded {fname}: {len(chunks)} chunks")

    print("Vectorstore build complete.")
    return col


def build_rl_actions_from_dict(output_dict: Dict[str, Any]) -> List[Dict[str, str]]:
    week_pattern = re.compile(r"(\d+)week$")
    actions_with_week: List[Dict[str, Any]] = []

    for key, value in output_dict.items():
        m = week_pattern.fullmatch(key)
        if not m:
            continue

        week = int(m.group(1))
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            continue

        var_name, delta = value
        baseline = output_dict.get(var_name)
        target = None
        try:
            if baseline is not None and delta is not None:
                target = float(baseline) + float(delta)
        except (TypeError, ValueError):
            pass

        actions_with_week.append(
            {
                "week": week,
                "variable": str(var_name),
                "baseline": baseline,
                "delta": delta,
                "target": target,
            }
        )

    actions_with_week.sort(key=lambda x: x["week"])

    rl_actions: List[Dict[str, str]] = []
    for a in actions_with_week:
        rl_actions.append(
            {
                "variable": a["variable"],
                "baseline": "" if a["baseline"] is None else str(a["baseline"]),
                "delta": "" if a["delta"] is None else str(a["delta"]),
                "target": "" if a["target"] is None else str(a["target"]),
            }
        )
    return rl_actions


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
        "You speak directly to the patient in clear, everyday language. "
        "Use short sentences. Avoid terms like 'glycemic control', 'cardiometabolic', or 'dyslipidemia'. "
        "Instead, say things like 'blood sugar', 'heart and blood vessels', or 'cholesterol and fats in the blood'. "
        "You acknowledge their real-life constraints (work schedule, money, mobility, stress) and say out loud that "
        "some changes may be hard, then suggest realistic options anyway.\n\n"
        "For EACH RL action, you receive these fields:\n"
        "- 'variable': name of the measure to change (for example 'glu', 'hb1ac', 'mvpa', 'bmi', or another label).\n"
        "- 'baseline': the current value (as a string).\n"
        "- 'delta': the suggested change (target minus baseline). A negative number means the goal is to LOWER the variable. "
        "A positive number means the goal is to RAISE the variable.\n"
        "- 'target': the desired value after the change (as a string).\n\n"
        "You do NOT need to know the exact clinical meaning of every variable name. If you recognize it (for example, as blood sugar, "
        "activity, weight, blood pressure, cholesterol, etc.), you can use that knowledge. If you do not recognize it, treat it as "
        "'this health measure' and still suggest reasonable, ADA-consistent behavior changes that could plausibly move it in the "
        "desired direction (up or down), such as changes in food, movement, sleep, stress, or substance use.\n\n"
        "For EACH RL action, you must:\n"
        "- Address the patient as 'you'.\n"
        "- Be encouraging, non-judgmental, and practical.\n"
        "- Use the sign of 'delta' and the values of 'baseline' and 'target' to understand what direction of change is needed.\n"
        "- Be specific about behaviors (e.g., 'walk briskly for 10 minutes', 'limit sugary drinks at dinner', 'do 2 sets of 10 sit-to-stands').\n"
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
            "1) A short heading like: '1. Working on glu: from 204 to 199', or "
            "'1. Working on [variable]: from [baseline] to [target]'.\n\n"
            "Then four labeled parts:\n"
            "- 'What this step means:' One short explanation of what you are asking the patient to do, "
            "clearly stating whether the goal is to increase or decrease the variable based on 'delta', and using 'baseline' and 'target' "
            "to describe the change in simple words.\n"
            "- 'Why this helps your health:' 1–2 simple sentences about how moving this measure in the right direction can support "
            "their health over time (for example, helping their body work better, lowering future risks, or helping them feel better day to day). "
            "Keep it general if you’re not sure what the variable means.\n"
            "- 'A realistic plan for you:' 3–4 sentences that:\n"
            "   * Start by acknowledging that this may be hard in their situation (for example, night shifts, little money, pain, housing), and\n"
            "   * Give 2–3 concrete actions with WHEN and HOW (for example, 'After your shift, walk 10 minutes from the bus stop', "
            "     'On days off, do 2 sets of 10 sit-to-stands from a chair while watching a video', "
            "     'During your break, swap one sweet drink for water'). The actions should be realistic and aimed at moving the variable "
            "        in the direction suggested by 'delta' (up or down).\n"
            "- 'How this fits ADA guidelines:' 1 short sentence saying this matches ADA diabetes lifestyle guidance, in plain language "
            "(for example, 'This fits ADA diabetes guidelines that encourage healthy daily habits to support your health over time.').\n\n"
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
        max_tokens=1500,
    )
    return resp.choices[0].message.content


# ----------------- PUBLIC API FUNCTION -----------------
def get_llm_recommendation(info: Dict[str, Any]) -> str:
    """
    End-to-end pipeline:
    - Builds/loads vectorstore
    - Parses clinical vars, context, and RL actions from `info`
    - Retrieves ADA evidence for each RL action
    - Calls LLM to generate recommendations
    - Returns a single text string with recommendation + evidence
    """
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found. Put it in a .env file.")

    collection = get_or_build_vectorstore()

    output_dict: Dict[str, Any] = info

    week_pattern = re.compile(r"\d+week$")

    clinical_vars = {
        k: v
        for k, v in output_dict.items()
        if k not in {"context", "old_score", "new_score"}
        and not week_pattern.fullmatch(k)
    }

    context = output_dict.get("context", "")
    clinical = json.dumps(clinical_vars, indent=2)

    rl_actions = build_rl_actions_from_dict(output_dict)

    age = clinical_vars.get("age")
    if isinstance(age, (int, float)) and age >= 65:
        pop = "older_adults"
    else:
        pop = "adults"

    evidence_blocks: List[Dict] = []
    for step in rl_actions:
        var = step["variable"]
        domain = None
        for key, dom in VARIABLE_TO_DOMAIN.items():
            if key in var.lower():
                domain = dom
                break
        if not domain:
            domain = "behaviors"

        q = f"ADA guideline for {var}, lifestyle change, diabetes, feasibility"
        ev = retrieve_guidelines(collection, domain, pop, q)
        evidence_blocks.append(
            {"variable": var, "domain": domain, "population": pop, "evidence": ev}
        )

    final_output = llm_generate(clinical, context, rl_actions, evidence_blocks)

    evidence_text_parts = []
    for i, ev in enumerate(evidence_blocks, start=1):
        block = ev["evidence"] or "(No evidence chunks retrieved)"
        evidence_text_parts.append(
            f"--- RL Action {i}: {ev['variable']} ---\n{block}"
        )
    evidence_text = "\n\n".join(evidence_text_parts)

    combined = (
        "FINAL RECOMMENDATION\n"
        "====================\n\n"
        f"{final_output}\n\n"
        "EVIDENCE USED (RAW ADA CHUNKS)\n"
        "==============================\n\n"
        f"{evidence_text}\n"
    )

    return combined


# ----------------- CLI ENTRYPOINT -----------------
def main():
    text = get_llm_recommendation(result)
    print(text)


if __name__ == "__main__":
    main()