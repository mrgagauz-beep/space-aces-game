import json
import sys
import pathlib

from jsonschema import validate, Draft7Validator


SCHEMA = {
    "type": "object",
    "properties": {
        "ROI": {
            "type": "object",
            "properties": {
                "MAIN": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "MINIMAP": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4,
                },
            },
            "required": ["MAIN", "MINIMAP"],
        }
    },
    "required": ["ROI"],
}


def main(path: str = "profiles/default.json") -> None:
    p = pathlib.Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    Draft7Validator.check_schema(SCHEMA)
    validate(instance=data, schema=SCHEMA)
    print("OK:", path)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
