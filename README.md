"""
filter.py — Diversity Filter for Activity and Contextual
=========================================================

Reads generated_concepts.json and filters Activity and Contextual categories
down to a fixed number of maximally diverse sentences per concept.

Baseline and Lexical are passed through unchanged.

Algorithm — Maximal Marginal Relevance (MMR) greedy selection:
  1. Encode all candidate sentences with LaBSE
  2. Seed with the sentence whose embedding has the largest L2 norm
  3. Iteratively add the sentence that minimises mean cosine similarity
     to the already-selected set
  4. Stop when TARGET count is reached

Contextual structural pre-filter (applied before MMR):
  - "the {concept}" must NOT appear in the first third of the sentence
  - Enforces the noise-first adverbial fronting design
  - If fewer than TARGET sentences pass, the concept is flagged and MMR
    runs on the full unfiltered pool as a fallback

Input:  data/processed/generated_concepts.json
Output: data/processed/filtered_concepts.json
"""

import os
import json
import torch
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────────────────────────
TARGET_ACTIVITY   = 8
TARGET_CONTEXTUAL = 8
NOUN_POS_MAX      = 0.33   # concept noun must appear after first 33% of sentence


# ── Structural pre-filter ─────────────────────────────────────────────────────
def passes_structure(sentence: str, concept: str) -> bool:
    """
    Returns True if 'the {concept}' appears after the first third of the sentence.
    Falls back to checking the bare concept noun if the full phrase is not found.
    """
    text   = sentence.lower()
    target = f"the {concept}".lower()
    pos    = text.find(target)
    if pos == -1:
        pos = text.find(concept.lower())
    if pos == -1:
        return False
    return (pos / len(sentence)) >= NOUN_POS_MAX


# ── MMR selection ─────────────────────────────────────────────────────────────
def mmr_select(model, sentences: list, k: int) -> list:
    """
    Greedily select k sentences from candidates to maximise pairwise diversity.

    Step 1 — encode and L2-normalise all sentences.
    Step 2 — seed with the sentence of largest raw embedding norm.
    Step 3 — iteratively pick the candidate with the lowest mean cosine
             similarity to the already-selected set.
    """
    if len(sentences) <= k:
        return sentences

    # Encode once
    raw_emb  = model.encode(sentences, convert_to_tensor=True, normalize_embeddings=False)
    norms    = torch.norm(raw_emb, dim=1)
    norm_emb = torch.nn.functional.normalize(raw_emb, dim=1)

    # Seed: sentence with largest embedding norm (most informative signal)
    seed_idx     = int(torch.argmax(norms).item())
    selected_idx = [seed_idx]
    remaining    = [i for i in range(len(sentences)) if i != seed_idx]

    while len(selected_idx) < k and remaining:
        best_idx   = None
        best_score = float("inf")

        for cand in remaining:
            # Mean cosine similarity from candidate to all already-selected
            sims     = [float(torch.dot(norm_emb[cand], norm_emb[s])) for s in selected_idx]
            mean_sim = sum(sims) / len(sims)
            if mean_sim < best_score:
                best_score = mean_sim
                best_idx   = cand

        selected_idx.append(best_idx)
        remaining.remove(best_idx)

    return [sentences[i] for i in selected_idx]


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    input_path  = os.path.join(project_root, "data", "processed", "generated_concepts.json")
    output_path = os.path.join(project_root, "data", "processed", "filtered_concepts.json")

    if not os.path.exists(input_path):
        print(f"Error: Cannot find input file at {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)

    print("Loading LaBSE model... (first run may take a while)")
    model = SentenceTransformer("sentence-transformers/LaBSE")
    print("Model loaded.\n")

    results = []
    flags   = []   # concepts where Contextual structural pre-filter fell short

    header = f"{'Concept':<24} {'Act in→out':<14} {'Ctx pre':<10} {'Ctx in→out'}"
    print(header)
    print("─" * 65)

    for item in concepts:
        concept    = item["concept"]
        categories = item["categories"]

        # Baseline and Lexical: pass through unchanged
        baseline_out = categories.get("Baseline", [])
        lexical_out  = categories.get("Lexical",  [])

        # ── Activity ──────────────────────────────────────────────────────────
        activity_in  = categories.get("Activity", [])
        activity_out = mmr_select(model, activity_in, TARGET_ACTIVITY)

        # ── Contextual: structural pre-filter then MMR ────────────────────────
        contextual_in       = categories.get("Contextual", [])
        contextual_struct   = [s for s in contextual_in if passes_structure(s, concept)]
        n_passed            = len(contextual_struct)

        if n_passed >= TARGET_CONTEXTUAL:
            contextual_out = mmr_select(model, contextual_struct, TARGET_CONTEXTUAL)
        else:
            # Fallback: run MMR on the full pool and flag for review
            contextual_out = mmr_select(model, contextual_in, TARGET_CONTEXTUAL)
            flags.append({
                "concept":           concept,
                "passed_structural": n_passed,
                "total":             len(contextual_in),
                "target":            TARGET_CONTEXTUAL,
            })

        print(f"{concept:<24} "
              f"{len(activity_in):>3} → {len(activity_out):<6}"
              f"{n_passed:>3}/{len(contextual_in):<6}"
              f"{len(contextual_in):>3} → {len(contextual_out)}")

        results.append({
            "concept": concept,
            "categories": {
                "Baseline":   baseline_out,
                "Lexical":    lexical_out,
                "Activity":   activity_out,
                "Contextual": contextual_out,
            }
        })

    # ── Write output ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print(f"Concepts processed : {len(results)}")
    print(f"Activity target    : {TARGET_ACTIVITY} per concept")
    print(f"Contextual target  : {TARGET_CONTEXTUAL} per concept")

    if flags:
        print(f"\n⚠  {len(flags)} concept(s) fell below the Contextual structural threshold:")
        for f_ in flags:
            print(f"   {f_['concept']:<22} "
                  f"passed {f_['passed_structural']}/{f_['total']} structural check "
                  f"(target {f_['target']}) — MMR run on full pool")
        print("   Consider regenerating these concepts.")
    else:
        print("\n✓  All concepts passed the Contextual structural check.")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()