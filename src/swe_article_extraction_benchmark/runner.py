import sys
from pathlib import Path

from running_stats.running_stats import RunningMeanVar

from swe_article_extraction_benchmark.data.loader import DataLoader
from swe_article_extraction_benchmark.data.saver import DataSaver
from swe_article_extraction_benchmark.evaluator import Evaluator
from swe_article_extraction_benchmark.extractors import BaseExtractor, ExtractorFactory


def evaluate_extractor(dataset_path: Path, extractor_name: str) -> None:

    try:
        dataset = DataLoader.load_jsonl(dataset_path)

        results_dir = Path("results")
        results_dir /= dataset.name
        results_dir.mkdir(parents=True, exist_ok=True)

        extractor: BaseExtractor = ExtractorFactory.create(extractor_name)
        evaluator = Evaluator()
        print(f"Evaluating extractor '{extractor.name}' ...", file=sys.stderr)
        print(f"   {extractor.name}: {extractor.description} ", file=sys.stderr)

        result = evaluator.evaluate(dataset=dataset, extractor=extractor)

        print(" Results")
        results_dict = result.to_dict()
        metrics = results_dict.get("overall_metrics", {})

        print(f"  overall: {metrics.get('overall', 0):.4f}")
        sample_results = results_dict.get("sample_results", [])
        if sample_results:
            extraction_times = [
                s.get("extraction_time", 0)
                for s in sample_results
                if s.get("extracion_success")
            ]
            if extraction_times:
                extracion_stats = RunningMeanVar()
                extracion_stats.push_iter(extraction_times)
                print(f"  average extraction time (s): {extracion_stats.mean():.3f}")
        success_count = sum(
            s for s in sample_results if s.get("extraction_success", False)
        )
        print(f"  sucess ratio: {success_count}/{len(dataset)}")
        extraction_results_path = (
            results_dir / f"{extractor.name}_html_extraction_results.jsonl"
        )
        print(f"Saving results to '{extraction_results_path}' ...")
        DataSaver.save_extraction_result(
            results=result,
            dataset=dataset,
            file_path=extraction_results_path,
            extractor_name=extractor.name,
        )
        results_path = (
            results_dir / f"{extractor.name}_html_extraction_evaluation_result.json"
        )
        report_path = (
            results_dir / f"{extractor.name}_html_extraction_evaluation_report.csv"
        )
        DataSaver.save_evaluation_results(result, results_path)
        DataSaver.save_summary_report(result, report_path)

        print("\n✅ Done")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


def compare_extractors(
    dataset_path: Path, extractor_names: list[str] | None = None
) -> None:
    print("\n=== Comparing extractors ===\n")

    results = {}
    extractors = ExtractorFactory.create_many_or_all(extractor_names)
    name_of_extractors = [e.name for e in extractors]
    print(f"{name_of_extractors=}")

    evaluator = Evaluator()
    dataset = DataLoader.load_jsonl(dataset_path)

    for extractor in extractors:
        print(f"Evaluating extractor: {extractor.name}", file=sys.stderr)
        print(f"  {extractor.name}: {extractor.description}", file=sys.stderr)

        try:
            result = evaluator.evaluate(dataset=dataset, extractor=extractor)

            results[result.extractor_name] = result

        except Exception as exc:
            print(f"Error evaluating {extractor.name}: {exc}", file=sys.stderr)
            continue

    print("RESULT")
    print("=" * 40)
    for extractor_name, result in results.items():
        overall_score = result.overall_metrics.get("overall", 0)
        print(f"{extractor_name}: {overall_score:.4f}")

    all_results = []
    for result in results.values():
        all_results.append(result.to_dict())

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True, parents=True)

    leaderboard_path = results_dir / "leaderboard.csv"
    evaluation_results_path = results_dir / "evaluation_results.json"
    dataset_with_results_path = results_dir / "dataset_with_results.jsonl"

    DataSaver.save_summary_report(all_results, leaderboard_path)
    DataSaver.save_evaluation_results(results, evaluation_results_path)
    DataSaver.save_extraction_result(
        results=all_results, dataset=dataset, file_path=dataset_with_results_path
    )

    print(f"\n📊 leaderboard saved to: {leaderboard_path}")
