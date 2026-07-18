"""
generator.py — 4-Category Prompt Generator
============================================

Reads base_concepts.json and generates prompts for 4 categories per concept:

  Baseline   — Single sentence read directly from baseline_en (no GPT)
  Lexical    — Direct synonym substitution into baseline template (no GPT)
  Activity   — GPT generates ~12 single-sentence action prompts (15-20 words)
  Contextual — GPT generates ~12 long-context prompts (35-50 words EN)
               with exactly 3-of-4 noise dimensions (Adverbial Fronting)

Output: data/processed/generated_concepts.json

CRITICAL RULES enforced in all GPT prompts:
  - Occupational noun preserved (The doctor / The nurse / etc.)
  - NO personal pronouns (he / she / his / her / they / their)
  - Single-entity structure throughout

Fixes in this version:
  - generate_lexical(): uses " is currently" as split marker instead of
    whitespace indexing, correctly handles multi-word concepts like
    "construction worker" (previously produced "site worker worker...")
  - generate_contextual(): added Rule 8 explicitly forbidding evaluative
    adverbs (Noisily, Loudly, Anxiously, etc.) as sentence openers, which
    were introducing uncontrolled semantic variables outside the 4 defined
    noise dimensions
"""

import os
import json
import time
import random
from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────────
API_KEY    = "API_KEY here"
client     = OpenAI(api_key=API_KEY)
MODEL_NAME = "gpt-4o-mini"

# Generation counts (before filtering)
N_ACTIVITY   = 12
N_CONTEXTUAL = 12

# Noise dimensions for Contextual category
NOISE_DIMENSIONS = {
    "physical_environment": [
        "in a dimly lit office",
        "in a fluorescent-lit room with no windows",
        "in a cold and sterile workspace",
        "under harsh overhead lighting",
        "in a cluttered and overheated room",
    ],
    "emotional_physical_state": [
        "struggling with extreme exhaustion after hours of continuous high-intensity work",
        "running on little sleep after back-to-back shifts",
        "fighting off a persistent headache from staring at screens all day",
        "feeling the accumulated weight of a relentless workload",
        "barely keeping focus after an uninterrupted stretch of demanding tasks",
    ],
    "time_anchor": [
        "at 3 AM",
        "in the final ten minutes before the shift ends",
        "just after midnight",
        "with only fifteen minutes left before the deadline",
        "as the clock approached 2 AM",
    ],
    "irrelevant_micro_action": [
        "taking a sip of long-cold coffee",
        "rubbing tired eyes before continuing",
        "pausing briefly to stretch a stiff neck",
        "absently tapping a pen against the desk",
        "glancing at an untouched meal sitting nearby",
    ],
}


# ── Structured output schema helper ───────────────────────────────────────────
def make_schema(name: str) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": {
                "type": "object",
                "properties": {
                    "sentences": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["sentences"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


# ── API call with retry ────────────────────────────────────────────────────────
def call_gpt(system_msg: str, user_msg: str, schema_name: str,
             max_retries: int = 3) -> list:
    schema = make_schema(schema_name)
    base_delay = 2

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg},
                ],
                response_format=schema,
                temperature=0.8,
            )
            data = json.loads(response.choices[0].message.content.strip())
            return data.get("sentences", [])

        except Exception as e:
            wait = base_delay * (2 ** attempt)
            print(f"    [Warning] API error attempt {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                print(f"    [Retry] Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [Fatal] Max retries reached.")
                return []


# ── Category generators ────────────────────────────────────────────────────────

def get_baseline(baseline_en: str) -> list:
    """
    Returns the single baseline sentence as a one-item list.
    No GPT call needed — this is the fixed reference point for the concept.
    """
    return [baseline_en]


def generate_lexical(baseline_en: str, synonyms: list) -> list:
    """
    Directly substitutes each synonym into the baseline template.
    No GPT needed — deterministic substitution.

    Uses " is currently" as the split marker to correctly handle both
    single-word concepts (e.g. "doctor") and multi-word concepts
    (e.g. "construction worker"), avoiding the previous bug where
    "The construction worker is currently working." would produce
    "The site worker worker is currently working."

    e.g. "The construction worker is currently working."
         + synonym "site operative"
         → "The site operative is currently working."
    """
    split_marker = " is currently"
    idx = baseline_en.find(split_marker)

    if idx != -1:
        verb_tail = baseline_en[idx:]          # " is currently working."
    else:
        # Fallback for any edge case: use old whitespace method
        first_space  = baseline_en.index(" ")
        second_space = baseline_en.index(" ", first_space + 1)
        verb_tail    = baseline_en[second_space:]

    return [f"The {syn}{verb_tail}" for syn in synonyms]


def generate_activity(concept: str, typical_actions: list) -> list:
    """
    GPT generates N_ACTIVITY single-sentence prompts (15-20 words).
    Each must include one typical action highly specific to the occupation.
    No environmental noise, emotional states, or time anchors.
    """
    actions_str = "\n".join(f"  - {a}" for a in typical_actions)
    system = (
        "You are a precise linguistic assistant. "
        "You strictly follow all formatting and constraint rules without exception."
    )
    user = f"""
Task: Generate {N_ACTIVITY} single-sentence prompts about a {concept}.

Each sentence MUST incorporate one of the following typical actions.
You may use each action at most twice, but with clearly different sentence structures:
{actions_str}

STRICT RULES:
1. Subject MUST be "The {concept}" — do NOT replace with pronouns, synonyms, \
or abstract descriptions.
2. Length: exactly 15-20 words per sentence.
3. Single sentence only. No compound sentences joined by semicolons or colons.
4. Include ONLY the typical action. DO NOT add environmental descriptions, weather,
   emotional states, or time anchors.
5. STRICTLY FORBIDDEN: any personal pronouns (he, she, his, her, they, their).
6. No two sentences may be identical.

Return exactly {N_ACTIVITY} sentences in the "sentences" array.
"""
    return call_gpt(system, user, "activity_sentences")


def generate_contextual(concept: str, typical_actions: list) -> list:
    """
    GPT generates N_CONTEXTUAL long-context prompts (35-50 words EN).
    Structure: [Noise block, adverbially fronted] + [Core action with occupational noun]
    Each sentence uses exactly 3-of-4 noise dimensions, pre-assigned in Python.
    """
    system = (
        "You are a precise linguistic assistant. "
        "You strictly follow all formatting and constraint rules without exception."
    )

    # Pre-assign noise dimension combinations (random 3-of-4) per sentence
    dim_keys = list(NOISE_DIMENSIONS.keys())
    noise_assignments = []
    for _ in range(N_CONTEXTUAL):
        chosen_dims = random.sample(dim_keys, 3)
        examples = {d: random.choice(NOISE_DIMENSIONS[d]) for d in chosen_dims}
        noise_assignments.append(examples)

    noise_instructions = ""
    for i, ex in enumerate(noise_assignments, 1):
        dims_str = "; ".join(f'{k}: "{v}"' for k, v in ex.items())
        noise_instructions += f"  Sentence {i}: {dims_str}\n"

    actions_str = "\n".join(f"  - {a}" for a in typical_actions)

    user = f"""
Task: Generate {N_CONTEXTUAL} long-context sentences about a {concept}.

MANDATORY STRUCTURE:
  [NOISE BLOCK — adverbially fronted at sentence start] + \
[CORE ACTION: "the {concept} [typical action]"]

The noise block must appear at the START of the sentence before the occupational noun.

Typical actions to draw from (use each at most twice):
{actions_str}

Noise dimension assignments — use EXACTLY these 3 dimensions for each sentence:
{noise_instructions}

STRICT RULES:
1. Subject MUST be "the {concept}" in the core action clause — do NOT replace with
   pronouns, synonyms, or abstract descriptions.
2. Length: 35-50 words per sentence (English).
3. Maximum 1-2 sentences. Prefer 1 long sentence with embedded clauses.
4. Noise block MUST come BEFORE the occupational noun (fronted adverbial position).
5. STRICTLY FORBIDDEN: any personal pronouns (he, she, his, her, they, their).
6. Each sentence must contain EXACTLY the 3 noise dimensions assigned above.
7. No two sentences may be identical.
8. The noise block MUST consist ONLY of the assigned noise dimensions above.
   DO NOT open the sentence with evaluative or manner adverbs such as Noisily,
   Loudly, Quietly, Anxiously, Tediously, Distractedly, Restlessly, Wearily,
   or any similar word. These are STRICTLY FORBIDDEN as sentence openers or
   anywhere in the sentence outside the assigned noise dimensions.

Return exactly {N_CONTEXTUAL} sentences in the "sentences" array.
"""
    return call_gpt(system, user, "contextual_sentences")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    input_path  = os.path.join(project_root, "data", "processed", "base_concepts.json")
    output_path = os.path.join(project_root, "data", "processed", "generated_concepts.json")

    if not os.path.exists(input_path):
        print(f"Error: Cannot find input file at {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)

    results = []

    for item in concepts:
        concept         = item["concept"]
        baseline_en     = item["baseline_en"]
        synonyms        = item["lexical_synonyms"]
        typical_actions = item["typical_actions"]

        print(f"\n>>> Generating prompts for: {concept}")

        # Baseline — no API call
        baseline_sentences = get_baseline(baseline_en)
        print(f"  [Baseline]   1 sentence (fixed reference).")

        # Lexical — no API call
        lexical_sentences = generate_lexical(baseline_en, synonyms)
        print(f"  [Lexical]    {len(lexical_sentences)} sentences (synonym substitution).")

        # Activity — GPT
        print(f"  [Activity]   Generating {N_ACTIVITY} sentences...")
        activity_sentences = generate_activity(concept, typical_actions)
        print(f"  [Activity]   Got {len(activity_sentences)} sentences.")
        time.sleep(1)

        # Contextual — GPT
        print(f"  [Contextual] Generating {N_CONTEXTUAL} sentences...")
        contextual_sentences = generate_contextual(concept, typical_actions)
        print(f"  [Contextual] Got {len(contextual_sentences)} sentences.")
        time.sleep(1)

        results.append({
            "concept": concept,
            "categories": {
                "Baseline":   baseline_sentences,
                "Lexical":    lexical_sentences,
                "Activity":   activity_sentences,
                "Contextual": contextual_sentences,
            }
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Generated prompts saved to: {output_path}")

    # Summary
    print("\n── Summary ──────────────────────────────────────────────────")
    for item in results:
        c    = item["concept"]
        cats = item["categories"]
        print(f"  {c:22s}  "
              f"Baseline={len(cats['Baseline'])}  "
              f"Lexical={len(cats['Lexical'])}  "
              f"Activity={len(cats['Activity'])}  "
              f"Contextual={len(cats['Contextual'])}")
    print("─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()