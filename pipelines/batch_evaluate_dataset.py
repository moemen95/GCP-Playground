"""GCP pipeline orchestration #2 — managed async BATCH evaluation.

Uses the v1beta1 ``evaluateDataset`` method (via the GenAI client / aiplatform
batch surface) to score a whole dataset asynchronously, instead of per-instance
``evaluateInstances``. Good for large golden sets / nightly regression runs.

Importable offline; calls require ``EVAL_BACKEND=vertex`` + ``[vertex]`` deps.
"""
from __future__ import annotations

import os


def require_vertex():
    if os.environ.get("EVAL_BACKEND") != "vertex":
        raise RuntimeError("set EVAL_BACKEND=vertex")
    try:
        import vertexai  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('requires `pip install -e ".[vertex]"`') from exc


def batch_evaluate(dataset_uri: str, *, project: str | None = None,
                   location: str = "us-central1", metrics: list[str] | None = None,
                   dest_uri: str | None = None):
    """Submit an async dataset-level evaluation job.

    ``dataset_uri``: a gs:// JSONL/CSV with prompt/response/reference/context columns.
    ``metrics``: metric ids, e.g. ['rouge_1','groundedness','question_answering_correctness'].
    Returns the long-running operation / result handle.
    """
    require_vertex()
    from vertexai import Client, types

    project = project or os.environ["GOOGLE_CLOUD_PROJECT"]
    metrics = metrics or ["rouge_1", "groundedness", "question_answering_correctness"]
    client = Client(project=project, location=location)

    # The GenAI client exposes batch evaluation over a dataset reference; the
    # underlying REST method is projects.locations.evaluateDataset (v1beta1).
    return client.evals.evaluate(
        dataset=types.EvaluationDataset(gcs_source=types.GcsSource(uris=[dataset_uri]))
        if hasattr(types, "EvaluationDataset") else dataset_uri,
        metrics=[types.Metric(name=m) for m in metrics],
        dest=dest_uri,
    )


if __name__ == "__main__":  # pragma: no cover
    try:
        print(batch_evaluate("gs://your-bucket/rag_qa.jsonl"))
    except (RuntimeError, KeyError) as exc:
        print(f"[batch_evaluate_dataset] requires vertex backend + project: {exc}")
