# Export OpenAPI spec from the FastAPI app
import json
import pathlib

from services.sheily_api.app import app

spec = app.openapi()
p = pathlib.Path(__file__).resolve().parents[1] / "openapi.json"
p.write_text(json.dumps(spec, indent=2), encoding="utf-8")
print("Wrote OpenAPI spec to", p)
