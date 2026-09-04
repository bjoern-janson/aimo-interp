# YOLO result custody

`YOLO001-B1.results.jsonl` must not contain a row before B1 closure. After closure, result events are append-only and never rewrite the frozen protocol or custody manifest.
