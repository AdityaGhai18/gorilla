import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Callable, Any

# ...small dynamic import helper to locate background module in same dir...
def _load_background_module():
	# attempt normal import first
	try:
		import background as bg
		return bg
	except Exception:
		# load by path relative to this file
		import importlib.util
		mod_path = Path(__file__).parent / "background.py"
		spec = importlib.util.spec_from_file_location("background", str(mod_path))
		mod = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(mod)
		return mod

bg = _load_background_module()

def generate_variants(
	audio_paths: List[str],
	effects_spec: Dict[str, Dict],
	output_root: str = "pipeline_outputs"
) -> List[Dict[str, Any]]:
	"""
	For each input audio and each effect in effects_spec, create one or more variants.
	effects_spec format:
	{
	  "effect_key": {
	    "fn": "apply_reverb",              # function name in background.py
	    "variants": [                      # list of kwargs dicts passed to fn (excluding audio_path/output_path)
	      {"mode":"cave"},
	      {"mode":"echo","echo_delay_ms":80,"echo_decay":0.5},
	      ...
	    ]
	  },
	  ...
	}
	Returns list of records: {audio_path, effect_key, variant_idx, variant_path, params}
	"""
	out_records = []
	out_root = Path(output_root)
	for audio_path in audio_paths:
		audio_base = Path(audio_path).stem
		for effect_key, spec in effects_spec.items():
			fn_name = spec.get("fn")
			variants = spec.get("variants", [{}])
			# resolve function
			if not hasattr(bg, fn_name):
				raise ValueError(f"Effect function not found in background.py: {fn_name}")
			fn = getattr(bg, fn_name)
			for i, params in enumerate(variants):
				# build output path
				subdir = out_root / audio_base / effect_key
				subdir.mkdir(parents=True, exist_ok=True)
				variant_fname = f"{audio_base}_{effect_key}_v{i}.mp3"
				out_path = subdir / variant_fname
				# call effect function (kwargs passed; audio_path and output_path assumed)
				# make a shallow copy of params to avoid mutation
				kwargs = dict(params)
				# ensure output_path param name; try common names
				if "output_path" in fn.__code__.co_varnames:
					kwargs["output_path"] = str(out_path)
					fn(audio_path, **kwargs)
				else:
					# fallback: call with audio_path, out_path, **kwargs
					fn(audio_path, str(out_path), **kwargs)
				rec = {
					"audio_path": audio_path,
					"audio_base": audio_base,
					"effect_key": effect_key,
					"variant_idx": i,
					"variant_path": str(out_path),
					"params": params
				}
				out_records.append(rec)
	return out_records

def run_evaluation(
	variant_records: List[Dict[str, Any]],
	llm_analysis_fn: Callable[[str, str, Dict[str, Any]], Dict[str, Any]],
	queries_map: Dict[str, str]
) -> List[Dict[str, Any]]:
	"""
	Run the provided llm_analysis_fn on each variant.
	llm_analysis_fn signature: (variant_audio_path, original_query_text, metadata) -> dict result
	queries_map maps audio_base -> original_query_text (the same subset of queries).
	Returns list of result records merging variant record + llm result.
	"""
	results = []
	for rec in variant_records:
		audio_base = rec["audio_base"]
		variant_path = rec["variant_path"]
		orig_query = queries_map.get(audio_base, "")
		# metadata includes effect params for the LLM if needed
		metadata = {"effect_key": rec["effect_key"], "params": rec["params"], "variant_idx": rec["variant_idx"]}
		# call user-supplied analysis function (may call LLM, ASR, classifier, etc.)
		analysis = llm_analysis_fn(variant_path, orig_query, metadata)
		out = dict(rec)
		out["analysis"] = analysis
		results.append(out)
	return results

def analyze_and_save(results: List[Dict[str, Any]], out_dir="pipeline_results"):
	"""
	Aggregate results to show which effects/variants cause which error labels.
	Assumes analysis contains either 'error' (bool) or 'error_labels' (list) or 'label' (str).
	Saves summary JSON and CSV per-effect.
	Returns path to summary JSON.
	"""
	out_path = Path(out_dir)
	out_path.mkdir(parents=True, exist_ok=True)
	summary = {"by_effect": {}}
	for r in results:
		effect = r["effect_key"]
		summary["by_effect"].setdefault(effect, {"total": 0, "errors": {}, "variants": {}})
		summary["by_effect"][effect]["total"] += 1
		analysis = r.get("analysis", {})
		# derive labels
		labels = []
		if isinstance(analysis, dict):
			if "error_labels" in analysis:
				labels = analysis["error_labels"]
			elif "label" in analysis:
				labels = [analysis["label"]]
			elif "error" in analysis and analysis["error"]:
				labels = ["error"]
		elif isinstance(analysis, str):
			labels = [analysis]
		# record labels counts
		for lab in labels:
			summary["by_effect"][effect]["errors"].setdefault(lab, 0)
			summary["by_effect"][effect]["errors"][lab] += 1
		# per variant record
		vk = f"v{r['variant_idx']}"
		vars_dict = summary["by_effect"][effect]["variants"]
		vars_dict.setdefault(vk, {"count": 0, "errors": {}})
		vars_dict[vk]["count"] += 1
		for lab in labels:
			vars_dict[vk]["errors"].setdefault(lab, 0)
			vars_dict[vk]["errors"][lab] += 1

	# write full results and summary
	(Path(out_dir) / "full_results.json").write_text(json.dumps(results, indent=2))
	(Path(out_dir) / "summary.json").write_text(json.dumps(summary, indent=2))

	# Also produce a flat CSV: effect,variant,variant_path,label
	csv_file = Path(out_dir) / "flat_results.csv"
	with open(csv_file, "w", newline="", encoding="utf-8") as fh:
		writer = csv.writer(fh)
		writer.writerow(["effect", "variant_idx", "variant_path", "labels", "raw_analysis"])
		for r in results:
			labels = []
			a = r.get("analysis", {})
			if isinstance(a, dict):
				if "error_labels" in a:
					labels = a["error_labels"]
				elif "label" in a:
					labels = [a["label"]]
				elif "error" in a and a["error"]:
					labels = ["error"]
			elif isinstance(a, str):
				labels = [a]
			writer.writerow([r["effect_key"], r["variant_idx"], r["variant_path"], ";".join(labels), json.dumps(a)])

	return str(Path(out_dir) / "summary.json")

def run_pipeline(
	audio_paths: List[str],
	queries_map: Dict[str, str],
	effects_spec: Dict[str, Dict],
	llm_analysis_fn: Callable[[str, str, Dict[str, Any]], Dict[str, Any]],
	output_root: str = "pipeline_outputs",
	results_dir: str = "pipeline_results"
) -> str:
	"""
	High-level helper: generate variants, run LLM evaluations, analyze and save summary.
	Returns summary json path.
	"""
	print("Generating variants...")
	vars_recs = generate_variants(audio_paths, effects_spec, output_root=output_root)
	print(f"Generated {len(vars_recs)} variants.")
	print("Running LLM evaluations...")
	results = run_evaluation(vars_recs, llm_analysis_fn, queries_map)
	print("Analyzing and saving results...")
	summary_path = analyze_and_save(results, out_dir=results_dir)
	print(f"Pipeline complete. Summary saved to: {summary_path}")
	return summary_path

def example_llm_analysis_fn(variant_audio_path: str, original_query_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Example llm_analysis_fn to pass into run_pipeline.
	- Tries to call a global transcribe_audio(path) if available (user should provide one).
	- Computes a simple WER between original_query_text and the transcript.
	- Returns a dict with keys: transcript, wer, error (bool), error_labels (list).
	Use this as a template; replace the transcription call with your ASR/LLM pipeline
	or wrap this function to call an LLM classifier on the transcript+metadata.
	"""
	# Attempt to transcribe using user-provided function if present
	transcript = None
	if "transcribe_audio" in globals() and callable(globals()["transcribe_audio"]):
		try:
			transcript = globals()["transcribe_audio"](variant_audio_path)
		except Exception as e:
			return {"label": "transcription_failed", "error": True, "reason": str(e)}

	# If no transcription available, return a diagnostic label
	if transcript is None:
		return {"label": "no_transcript", "error": True, "reason": "no transcribe_audio() available"}

	# simple WER implementation
	def _wer(ref: str, hyp: str) -> float:
		r = ref.split()
		h = hyp.split()
		n = len(r)
		if n == 0:
			return 0.0 if len(h) == 0 else 1.0
		# DP matrix
		d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
		for i in range(len(r) + 1):
			d[i][0] = i
		for j in range(len(h) + 1):
			d[0][j] = j
		for i in range(1, len(r) + 1):
			for j in range(1, len(h) + 1):
				sub_cost = 0 if r[i - 1] == h[j - 1] else 1
				d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + sub_cost)
		edits = d[len(r)][len(h)]
		return edits / max(1, n)

	wer_score = _wer((original_query_text or "").lower(), transcript.lower())
	labels = []
	if wer_score > 0.4:
		labels.append("high_wer")
	elif wer_score > 0.15:
		labels.append("medium_wer")
	else:
		labels.append("low_wer")
	# small heuristic label examples based on metadata
	if metadata and "effect_key" in metadata and "noise" in str(metadata["effect_key"]).lower():
		labels.append("noisy_effect")

	return {
		"transcript": transcript,
		"wer": wer_score,
		"error": wer_score > 0.15,
		"error_labels": labels
	}

# new helper: generate dataset + manifest for LLM consumption
def generate_dataset_for_llm(
    audio_paths: List[str],
    queries_map: Dict[str, str],
    effects_spec: Dict[str, Dict],
    output_root: str = "pipeline_outputs",
    manifest_csv: str = "pipeline_for_llm.csv",
    manifest_json: str = "pipeline_for_llm.json",
    zip_output: bool = False,
    zip_name: str = "pipeline_outputs_zip"
) -> Dict[str, str]:
    """
    Generate all variants and write a manifest (CSV + JSON) that maps each variant file
    to the original query text and effect metadata so you can feed the audio files
    into your LLM/ASR pipeline.

    Returns a dict with paths: {"manifest_csv": ..., "manifest_json": ..., "output_root": ..., "zip": ... (optional)}
    """
    from pathlib import Path
    import json as _json
    import csv as _csv
    import shutil as _shutil

    # 1) generate variants (writes files under output_root)
    records = generate_variants(audio_paths, effects_spec, output_root=output_root)

    # 2) build manifest entries
    manifest_rows = []
    for rec in records:
        audio_base = rec.get("audio_base")
        orig_query = queries_map.get(audio_base, "")
        row = {
            "audio_base": audio_base,
            "effect_key": rec.get("effect_key"),
            "variant_idx": rec.get("variant_idx"),
            "variant_path": rec.get("variant_path"),
            "original_query_text": orig_query,
            "params": rec.get("params", {})
        }
        manifest_rows.append(row)

    # 3) write JSON manifest
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_json_path = out_dir / manifest_json
    manifest_json_path.write_text(_json.dumps(manifest_rows, indent=2))

    # 4) write CSV manifest (flat, easier to ingest)
    manifest_csv_path = out_dir / manifest_csv
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = _csv.writer(fh)
        writer.writerow(["audio_base", "effect_key", "variant_idx", "variant_path", "original_query_text", "params_json"])
        for r in manifest_rows:
            writer.writerow([
                r["audio_base"],
                r["effect_key"],
                r["variant_idx"],
                r["variant_path"],
                r["original_query_text"],
                _json.dumps(r["params"], ensure_ascii=False)
            ])

    result = {
        "manifest_csv": str(manifest_csv_path),
        "manifest_json": str(manifest_json_path),
        "output_root": str(out_dir)
    }

    # 5) optional zip of the output_root for upload
    if zip_output:
        zip_path = str(out_dir.parent / zip_name)
        # shutil.make_archive will append .zip
        _shutil.make_archive(zip_path, 'zip', root_dir=str(out_dir))
        result["zip"] = f"{zip_path}.zip"

    return result

def example_llm_judge_wrapper(variant_audio_path: str, original_query_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Small adapter so example_llm_analysis_fn can be used as a simple judge.
	Replace this wrapper with a function that calls your LLM and returns a dict:
	{
	  "error_types": ["high_wer", "misheard_entity"],
	  "transcript": "...",
	  "notes": "optional free text"
	}
	"""
	# use example_llm_analysis_fn (which returns wer/labels) as a fallback judge
	analysis = example_llm_analysis_fn(variant_audio_path, original_query_text, metadata)
	labels = analysis.get("error_labels", [])
	# map generic labels to error_types expected by the experiment
	error_types = []
	if "high_wer" in labels or "medium_wer" in labels:
		error_types.append("transcription_error")
	if "noisy_effect" in labels:
		error_types.append("background_confusion")
	# include raw transcript if present
	return {"error_types": error_types or ["no_error"], "transcript": analysis.get("transcript"), "raw": analysis}

def run_full_experiment(
	audio_dir: str,
	queries_map: Dict[str, str],
	effects_spec: Dict[str, Dict],
	llm_judge_fn: Callable[[str, str, Dict[str, Any]], Dict[str, Any]] = None,
	transcribe_fn: Callable[[str], str] = None,
	output_root: str = "pipeline_outputs",
	results_dir: str = "pipeline_results",
	audio_glob: str = "*.wav"
) -> str:
	"""
	High-level experiment runner:
	1) Finds original audio files under audio_dir matching audio_glob.
	2) Generates controlled noisy variants using effects_spec (calls generate_dataset_for_llm).
	3) Runs llm_judge_fn on each generated variant to obtain error labels/types.
	4) Aggregates correlations between effect_key and error types and saves results.

	llm_judge_fn signature: (variant_audio_path, original_query_text, metadata) -> dict
	  Expected dict keys: "error_types" (list of strings) and optional "transcript"/"notes".
	If llm_judge_fn is None, example_llm_judge_wrapper is used (which in turn uses example_llm_analysis_fn
	and requires transcribe_audio to be defined in the runtime).
	"""
	from pathlib import Path
	import csv as _csv

	# 1) find audio files
	root = Path(audio_dir)
	if not root.exists():
		raise FileNotFoundError(f"audio_dir not found: {audio_dir}")
	audio_paths = [str(p) for p in root.rglob(audio_glob)]

	if len(audio_paths) == 0:
		# try common alternatives
		audio_paths = []
		for ext in ("*.wav", "*.mp3", "*.flac"):
			audio_paths.extend([str(p) for p in root.rglob(ext)])
	if len(audio_paths) == 0:
		raise RuntimeError(f"No audio files found under {audio_dir} (tried {audio_glob} and common extensions)")

	# 2) generate variants & manifest
	gen_res = generate_dataset_for_llm(audio_paths, queries_map, effects_spec, output_root=output_root)
	manifest_path = Path(gen_res["manifest_json"])
	manifest = json.loads(manifest_path.read_text())

	# 3) choose judge fn
	# judge_fn = llm_judge_fn or example_llm_judge_wrapper
	judge_fn = llm_judge_fn

	# 4) iterate and call judge, collect raw judgments
	results = []
	for rec in manifest:
		variant_path = rec["variant_path"]
		orig_query = rec.get("original_query_text", "")
		metadata = {"effect_key": rec.get("effect_key"), "params": rec.get("params"), "variant_idx": rec.get("variant_idx"), "audio_base": rec.get("audio_base")}
		try:
			judg = judge_fn(variant_path, orig_query, metadata)
		except Exception as e:
			judg = {"error": True, "error_types": ["judge_failed"], "notes": str(e)}
		entry = dict(rec)
		entry["judgment"] = judg
		results.append(entry)

	# 5) save raw results
	out_dir = Path(results_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	(Path(out_dir) / "full_judgments.json").write_text(json.dumps(results, indent=2))

	# 6) create flat CSV for review
	csv_path = Path(out_dir) / "flat_judgments.csv"
	with open(csv_path, "w", newline="", encoding="utf-8") as fh:
		writer = _csv.writer(fh)
		writer.writerow(["audio_base", "effect_key", "variant_idx", "variant_path", "original_query_text", "error_types", "transcript", "notes"])
		for r in results:
			j = r.get("judgment", {})
			errs = j.get("error_types", []) if isinstance(j, dict) else []
			writer.writerow([r.get("audio_base"), r.get("effect_key"), r.get("variant_idx"), r.get("variant_path"), r.get("original_query_text"), ";".join(errs), j.get("transcript") if isinstance(j, dict) else None, json.dumps(j)])

	# 7) aggregate correlations: effect_key -> error_type counts and reverse
	summary = {"by_effect": {}, "by_error_type": {}}
	for r in results:
		effect = r.get("effect_key")
		j = r.get("judgment", {}) or {}
		err_list = j.get("error_types", ["no_error"]) if isinstance(j, dict) else ["no_judgment"]
		summary["by_effect"].setdefault(effect, {"total": 0, "error_counts": {}})
		summary["by_effect"][effect]["total"] += 1
		for e in err_list:
			summary["by_effect"][effect]["error_counts"].setdefault(e, 0)
			summary["by_effect"][effect]["error_counts"][e] += 1

			summary["by_error_type"].setdefault(e, {"total": 0, "effects": {}})
			summary["by_error_type"][e]["total"] += 1
			summary["by_error_type"][e]["effects"].setdefault(effect, 0)
			summary["by_error_type"][e]["effects"][effect] += 1

	# 8) save summary
	(Path(out_dir) / "experiment_summary.json").write_text(json.dumps(summary, indent=2))

	# 9) Return path to summary for easy consumption
	return str(Path(out_dir) / "experiment_summary.json")

if __name__ == "__main__":
	# Example usage of run_full_experiment
	audio_directory = "berkeley-function-call-leaderboard/audio_calling/audio/BFCL_v3_live_simple"
	# Example queries map: audio_base -> original query text
	queries = {
		"audio1": "What is the weather today?",
		"audio2": "Play some music.",
		# ...
	}
	# Example effects specification
	import judge
	judge_fn = judge.create_llm_judge_function()
	effects = {
		"reverb": {
			"fn": "apply_reverb",
			"variants": [{"mode": "cave"}, {"mode": "hall"}]
		},
		"background_noise": {
			"fn": "add_background_noise",
			"variants": [{"noise_level": 0.2}, {"noise_level": 0.5}]
		}
	}
	summary_path = run_full_experiment(
		audio_dir=audio_directory,
		queries_map=queries,
		effects_spec=effects,
		llm_judge_fn=judge_fn,
		output_root="pipeline_outputs",
		results_dir="pipeline_results",
		audio_glob="*.mp3"
	)
	print(f"Experiment summary saved to: {summary_path}")