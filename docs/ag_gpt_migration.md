# ag-gpt integration boundary

GeoFarmAI is the reusable scientific management-zone library. The separate
ag-gpt repository owns conversational, web, application-state, visualization,
and analysis-planning responsibilities and imports GeoFarmAI through its public
API. GeoFarmAI never imports ag-gpt.

## Phase 9B separation

The ag-gpt migration is complete. The following application implementations
were removed from the GeoFarmAI source tree and distribution:

- analysis planning and background job state;
- application analysis and artifact registries;
- controlled user-code sandbox execution;
- project/workspace manifests;
- interactive-map and result-presentation services;
- Prefect-based historical workflow jobs;
- the old ag-gpt-specific two-CSV terminal runner.

Equivalent application behavior is maintained in ag-gpt. Scientific execution
uses `FieldDataset`, `GeoFarmModel`, `GeoFarmResult`, `delineate_zones`,
`load_config`, and `run_from_config`.

## Retained scientific compatibility

The top-level `core` package remains temporarily because canonical GeoFarmAI
uses its established clustering, evaluation, decomposition, harmonization,
spatial, raster, and export implementations. Moving those algorithms into the
`geofarmai` namespace is a separate namespace-cleanup phase and must preserve
scientific behavior.

`run_pipeline(data, **model_options)` remains as a deprecated, purely
scientific wrapper over `GeoFarmModel.fit`. Mapping-based application workflow
orchestration is no longer supported; use `run_from_config(path, legacy=True)`
for a historical YAML file.

## Distribution boundary

The wheel contains only:

- `geofarmai/` — canonical public scientific APIs;
- `core/` — retained scientific implementations and compatibility utilities;
- distribution metadata.

It does not contain `services/`, `jobs/`, an LLM integration, a server, or a
web interface.
