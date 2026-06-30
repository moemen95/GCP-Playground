"""GCP pipeline orchestration #1 — Vertex AI Pipelines (Kubeflow / KFP).

Wraps the Gen AI evaluation as a schedulable Vertex AI Pipeline DAG so evals run
as managed jobs (tracked in Vertex AI Experiments) rather than ad-hoc scripts.
AutoSxS itself ships as a managed pipeline template (``autosxs-template/<ver>``);
this module shows both (a) authoring a custom KFP component that calls
``client.evals.evaluate`` and (b) launching the AutoSxS template.

Importable offline; compiling/running needs ``pip install -e ".[pipelines,vertex]"``
and ``EVAL_BACKEND=vertex`` + a project. Every entrypoint guards with a clear
RuntimeError when the SDK/creds are absent.
"""
from __future__ import annotations

import os

PIPELINE_ROOT_ENV = "PIPELINE_ROOT"  # gs://bucket/pipeline-root


def _require(pkg: str):
    try:
        __import__(pkg)
    except ImportError as exc:  # pragma: no cover - offline guard
        raise RuntimeError(
            f"{pkg} not installed; run `pip install -e \".[pipelines,vertex]\"` "
            f"and set EVAL_BACKEND=vertex with a GCP project."
        ) from exc


def build_eval_pipeline():
    """Define a KFP pipeline: run_inference -> evaluate -> emit metrics artifact.

    Returns the (uncompiled) pipeline function. Compile with
    ``kfp.compiler.Compiler().compile(build_eval_pipeline(), 'eval_pipeline.json')``.
    """
    _require("kfp")
    from kfp import dsl

    @dsl.component(base_image="python:3.11",
                   packages_to_install=["google-cloud-aiplatform[evaluation]", "google-genai", "pandas"])
    def evaluate_component(project: str, location: str, dataset_uri: str,
                           metrics: list, experiment: str) -> dict:
        # Runs inside the pipeline step (so imports are local to the component).
        import pandas as pd
        from vertexai import Client, types  # noqa: F401

        client = Client(project=project, location=location)
        df = pd.read_json(dataset_uri, lines=True)
        result = client.evals.evaluate(
            dataset=df,
            metrics=[types.Metric(name=m) for m in metrics],
        )
        return {"summary": getattr(result, "summary_metrics", {})}

    @dsl.pipeline(name="card-benefits-eval", description="Gen AI eval of the benefits-finder agent")
    def pipeline(project: str, location: str = "us-central1",
                 dataset_uri: str = "gs://your-bucket/rag_qa.jsonl",
                 experiment: str = "card-benefits-eval"):
        evaluate_component(
            project=project, location=location, dataset_uri=dataset_uri,
            metrics=["rouge_1", "groundedness", "question_answering_correctness"],
            experiment=experiment,
        )

    return pipeline


def compile_pipeline(out_path: str = "eval_pipeline.json") -> str:
    _require("kfp")
    from kfp import compiler

    compiler.Compiler().compile(build_eval_pipeline(), out_path)
    return out_path


def run_pipeline(project: str | None = None, location: str = "us-central1",
                 pipeline_root: str | None = None, template_path: str = "eval_pipeline.json"):
    """Submit the compiled pipeline to Vertex AI Pipelines."""
    _require("google.cloud.aiplatform")
    from google.cloud import aiplatform

    project = project or os.environ["GOOGLE_CLOUD_PROJECT"]
    pipeline_root = pipeline_root or os.environ.get(PIPELINE_ROOT_ENV)
    aiplatform.init(project=project, location=location)
    job = aiplatform.PipelineJob(
        display_name="card-benefits-eval",
        template_path=template_path,
        pipeline_root=pipeline_root,
        parameter_values={"project": project, "location": location},
    )
    job.submit(experiment=os.environ.get("VERTEX_EXPERIMENT", "card-benefits-eval"))
    return job


def launch_autosxs(project: str, location: str = "us-central1", *,
                   evaluation_dataset: str, response_column_a: str = "response_a",
                   response_column_b: str = "response_b", task: str = "question_answering",
                   human_preference_column: str | None = None,
                   template_version: str = "2.8.0"):
    """Launch the managed AutoSxS pairwise pipeline (Cohen's-Kappa vs humans when
    ``human_preference_column`` is supplied)."""
    _require("google.cloud.aiplatform")
    from google.cloud import aiplatform
    from google_cloud_pipeline_components.v1 import model_evaluation  # noqa: F401

    aiplatform.init(project=project, location=location)
    template_uri = (
        f"https://us-kfp.pkg.dev/ml-pipeline/google-cloud-registry/autosxs-template/{template_version}"
    )
    params = {
        "evaluation_dataset": evaluation_dataset,
        "id_columns": ["id"],
        "task": task,
        "response_column_a": response_column_a,
        "response_column_b": response_column_b,
        "autorater_prompt_parameters": {
            "inference_instruction": {"column": "question"},
            "inference_context": {"column": "context"},
        },
    }
    if human_preference_column:
        params["human_preference_column"] = human_preference_column
    job = aiplatform.PipelineJob(
        display_name="card-benefits-autosxs",
        template_path=template_uri,
        parameter_values=params,
        pipeline_root=os.environ.get(PIPELINE_ROOT_ENV),
    )
    job.submit()
    return job


if __name__ == "__main__":  # pragma: no cover
    try:
        print("Compiling eval pipeline ->", compile_pipeline())
    except RuntimeError as exc:
        print(f"[vertex_pipeline] {exc}")
