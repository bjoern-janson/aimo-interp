from aimo_interp_infra.model_lifecycle import BatchModelExecutor


def test_executor_loads_exactly_once_for_one_batch():
    loads: list[str] = []
    releases: list[str] = []

    def loader(model_id: str) -> dict[str, str]:
        loads.append(model_id)
        return {"model_id": model_id}

    def processor(model: dict[str, str], problem: str) -> str:
        return f"{model['model_id']}::{problem}"

    def releaser(model: dict[str, str]) -> None:
        releases.append(model["model_id"])

    executor = BatchModelExecutor(loader, processor, releaser)
    output = executor.run("example/model", ["p1", "p2", "p3"])

    assert output == [
        "example/model::p1",
        "example/model::p2",
        "example/model::p3",
    ]
    assert loads == ["example/model"]
    assert releases == ["example/model"]


def test_executor_releases_model_when_processor_raises():
    released: list[str] = []

    def loader(model_id: str) -> dict[str, str]:
        return {"model_id": model_id}

    def processor(model: dict[str, str], problem: str) -> str:
        raise RuntimeError("synthetic processor failure")

    def releaser(model: dict[str, str]) -> None:
        released.append(model["model_id"])

    executor = BatchModelExecutor(loader, processor, releaser)

    try:
        executor.run("example/model", ["p1"])
    except RuntimeError as exc:
        assert "synthetic processor failure" in str(exc)
    else:
        raise AssertionError("processor failure must propagate")

    assert released == ["example/model"]

