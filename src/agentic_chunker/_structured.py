"""Helpers for user-defined structured extraction models."""
from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
import json
from typing import Any


@dataclass
class StructuredExtraction:
    """A user-defined structured domain record with source evidence.

    ``type`` is the user model name, not a core ontology type. ``data`` remains a
    plain dictionary so extraction results are serializable without requiring a
    runtime dependency on Pydantic or another validation library.
    """

    type: str
    data: dict[str, Any]
    evidence: str
    source_chunk_id: str
    metadata: dict = field(default_factory=dict)


def structured_from_json(raw: dict, source_chunk_id: str) -> list[StructuredExtraction]:
    items = raw.get("structured_extractions", [])
    if isinstance(items, dict):
        items = _expand_keyed_extractions(items)

    structured: list[StructuredExtraction] = []
    if not isinstance(items, list):
        return structured

    for item in items:
        if not isinstance(item, dict):
            continue
        extraction_type = item.get("type") or item.get("model") or item.get("name")
        if not isinstance(extraction_type, str) or not extraction_type:
            continue
        data = item.get("data")
        if not isinstance(data, dict):
            data = {
                key: value
                for key, value in item.items()
                if key not in {"type", "model", "name", "evidence", "source_chunk_id", "metadata"}
            }
        evidence = item.get("evidence")
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        structured.append(StructuredExtraction(
            type=extraction_type,
            data=dict(data),
            evidence=evidence if isinstance(evidence, str) else _evidence_from_data(data),
            source_chunk_id=source_chunk_id,
            metadata=metadata,
        ))
    return structured


def validate_structured_extractions(
    extractions: list[StructuredExtraction],
    models: list[type],
    source_chunk_id: str,
) -> list[StructuredExtraction]:
    model_by_name = {model_name(model): model for model in models}
    validated: list[StructuredExtraction] = []
    for extraction in extractions:
        if model_by_name and extraction.type not in model_by_name:
            continue
        model = model_by_name.get(extraction.type)
        data = _validated_model_data(model, extraction.data) if model else dict(extraction.data)
        if data is None:
            continue
        validated.append(StructuredExtraction(
            type=extraction.type,
            data=data,
            evidence=extraction.evidence or _evidence_from_data(data),
            source_chunk_id=extraction.source_chunk_id or source_chunk_id,
            metadata=dict(extraction.metadata),
        ))
    return validated


def model_specs(models: list[type]) -> list[dict]:
    return [_model_spec(model) for model in models]


def model_name(model: type) -> str:
    return getattr(model, "__name__", str(model))


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _expand_keyed_extractions(items: dict) -> list[dict]:
    expanded = []
    for model, model_items in items.items():
        if isinstance(model_items, list):
            for item in model_items:
                if isinstance(item, dict):
                    expanded.append({"type": str(model), "data": item})
    return expanded


def _model_spec(model: type) -> dict:
    spec: dict[str, Any] = {
        "name": model_name(model),
        "description": _clean_doc(getattr(model, "__doc__", "")),
        "fields": _model_field_specs(model),
    }
    schema = _json_schema(model)
    if schema:
        spec["json_schema"] = schema
    return spec


def _model_field_specs(model: type) -> list[dict]:
    if is_dataclass(model):
        return [
            {
                "name": f.name,
                "type": _type_name(f.type),
                "required": f.default is MISSING and f.default_factory is MISSING,
            }
            for f in fields(model)
        ]

    annotations = getattr(model, "__annotations__", {})
    if isinstance(annotations, dict) and annotations:
        return [
            {
                "name": name,
                "type": _type_name(annotation),
                "required": not hasattr(model, name),
            }
            for name, annotation in annotations.items()
        ]

    schema = _json_schema(model)
    properties = schema.get("properties", {}) if schema else {}
    required = set(schema.get("required", [])) if schema else set()
    if not isinstance(properties, dict):
        return []
    return [
        {
            "name": name,
            "type": _type_name(prop.get("type", "unknown")) if isinstance(prop, dict) else "unknown",
            "required": name in required,
        }
        for name, prop in properties.items()
    ]


def _json_schema(model: type) -> dict:
    for method_name in ("model_json_schema", "schema"):
        schema_fn = getattr(model, method_name, None)
        if not callable(schema_fn):
            continue
        try:
            schema = schema_fn()
        except Exception:
            schema = None
        return schema if isinstance(schema, dict) else {}
    return {}


def _validated_model_data(model: type | None, data: dict) -> dict | None:
    if model is None:
        return dict(data)

    validate = getattr(model, "model_validate", None)
    if callable(validate):
        try:
            return _object_to_dict(validate(data))
        except Exception:
            return None

    parse_obj = getattr(model, "parse_obj", None)
    if callable(parse_obj):
        try:
            return _object_to_dict(parse_obj(data))
        except Exception:
            return None

    if is_dataclass(model):
        try:
            allowed = {f.name for f in fields(model)}
            kwargs = {key: value for key, value in data.items() if key in allowed}
            return asdict(model(**kwargs))
        except Exception:
            return None

    annotations = getattr(model, "__annotations__", {})
    if isinstance(annotations, dict) and annotations:
        allowed = set(annotations)
        missing = [name for name in allowed if name not in data and not hasattr(model, name)]
        if missing:
            return None
        return {key: value for key, value in data.items() if key in allowed}

    return dict(data)


def _object_to_dict(value: Any) -> dict | None:
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        data = dump()
        return data if isinstance(data, dict) else None
    dict_fn = getattr(value, "dict", None)
    if callable(dict_fn):
        data = dict_fn()
        return data if isinstance(data, dict) else None
    return None


def _evidence_from_data(data: dict) -> str:
    evidence = data.get("evidence")
    return evidence if isinstance(evidence, str) else ""


def _clean_doc(text: str) -> str:
    return " ".join(text.split())


def _type_name(value: Any) -> str:
    return getattr(value, "__name__", str(value))
