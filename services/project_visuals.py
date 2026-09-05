from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

METRIC_ALIASES: Dict[str, List[str]] = {
    "vr": ["vr", "variance reduction", "yield variance", "yield variance reduction"],
    "ch_score": ["ch", "ch score", "calinski harabasz", "calinski-harabasz", "pseudo f", "pseudo-f"],
    "asc": ["asc", "silhouette", "average silhouette", "silhouette coefficient"],
    "anova_p": ["anova", "anova p", "anova p value", "anova p-value", "yield significance"],
}

VISUAL_TYPE_ALIASES: Dict[str, List[str]] = {
    "selected_map": ["map", "cluster map", "zone map", "management zone map", "map generated from", "best map"],
    "comparison_figure": ["graph", "chart", "comparison", "comparison figure", "score figure", "metric figure"],
    "interpolation_surface": ["interpolation", "interpolated", "kriging", "surface", "spatial surface"],
    "component_map": ["pc", "principal component", "component", "feature map"],
}

VARIABLE_ALIASES: Dict[str, List[str]] = {
    "Curve": ["curve", "curvature", "terrain curvature"],
    "EC_DP": ["ec", "electrical conductivity", "deep ec", "ec dp", "soil conductivity"],
    "IR": ["ir", "infrared"],
    "Moisture": ["moisture", "soil moisture", "wetness", "wetter", "water content"],
    "Red": ["red", "red band", "spectral red"],
    "Slope": ["slope", "terrain slope"],
    "Soil_Temp_C": ["soil temperature", "soil temp", "temperature surface"],
    "Yld_Mass_Dry_lb_ac": ["yield", "dry yield", "yield mass", "yield surface", "yield interpolation"],
}

ROLE_PATTERNS: Dict[str, str] = {
    "yield_variance_cluster_map": "clusters_best.png",
    "recommended_cluster_map": "clusters_best.png",
    "ch_cluster_map": "clusters_best_ch_score.png",
    "vr_comparison": "comparison_vr.png",
    "ch_score_comparison": "comparison_ch_score.png",
    "asc_comparison": "comparison_asc.png",
    "anova_comparison": "comparison_anova_p.png",
    "pca_component_1": "feature_PC1.png",
    "pca_component_2": "feature_PC2.png",
    "pca_component_3": "feature_PC3.png",
}

VISUAL_CAPTIONS: Dict[str, str] = {
    "yield_variance_cluster_map": "Outcome-variance-selected management-zone map. This map is selected using outcome variance reduction.",
    "recommended_cluster_map": "Recommended outcome-variance management-zone map for the field.",
    "ch_cluster_map": "Calinski-Harabasz-selected management-zone map. This map is selected using feature-space separation.",
    "vr_comparison": "Comparison of candidate models by outcome variance reduction.",
    "ch_score_comparison": "Comparison of candidate models by Calinski-Harabasz score.",
    "asc_comparison": "Comparison of candidate models by average silhouette coefficient.",
    "anova_comparison": "Comparison of candidate models by outcome ANOVA p-value.",
    "cluster_map_comparison": "Visual comparison of the yield-variance-selected map and the Calinski-Harabasz-selected map.",
}

MEANINGLESS_TERMS = {
    "a",
    "an",
    "and",
    "best",
    "can",
    "figure",
    "from",
    "generated",
    "image",
    "me",
    "of",
    "please",
    "result",
    "results",
    "show",
    "the",
    "to",
}


class VisualCatalogItem(BaseModel):
    id: str
    path: str
    filename: str
    title: str
    description: str
    category: str
    visual_type: str
    role: Optional[str] = None
    metric: Optional[str] = None
    variable: Optional[str] = None
    component: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)


class VisualQuery(BaseModel):
    metric: Optional[str] = None
    variable: Optional[str] = None
    component: Optional[str] = None
    visual_type: Optional[str] = None
    wants_multiple: bool = False
    wants_best: bool = False
    raw_text: str


class VisualMatch(BaseModel):
    item: VisualCatalogItem
    score: float
    reasons: List[str] = Field(default_factory=list)


class VisualResolution(BaseModel):
    status: Literal["resolved", "choices", "not_found"]
    selected_ids: List[str] = Field(default_factory=list)
    option_ids: List[str] = Field(default_factory=list)
    message: str


def normalize_visual_text(value: str) -> str:
    """Normalize request/catalog text for deterministic visual matching."""

    table = str.maketrans({char: " " for char in string.punctuation if char not in {"_", "-"} })
    value = value.lower().replace("_", " ").replace("-", " ").translate(table)
    return re.sub(r"\s+", " ", value).strip()


def resolve_output_dir(cfg: dict) -> Path:
    """Return the configured output directory."""

    return Path(cfg.get("export", {}).get("out_dir", "outputs")).resolve()


def resolve_project_raster_dir(cfg: dict) -> Path:
    """Return the project raster output directory for generated map products."""

    project_name = cfg.get("project", {}).get("name", "project")
    return resolve_output_dir(cfg) / f"{project_name}_raster"


def resolve_project_preview_dir(cfg: dict) -> Path:
    """Return the preview image directory for the configured project."""

    return resolve_project_raster_dir(cfg) / "preview"


def validate_visual_path(path: Path, output_dir: Path) -> Path:
    """Resolve and validate an image path inside the configured output directory."""

    resolved = path.resolve()
    output_resolved = output_dir.resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"Expected preview file was not found: {path}")
    try:
        resolved.relative_to(output_resolved)
    except ValueError as exc:
        raise ValueError(f"Visual path is outside the configured output directory: {path}") from exc
    if resolved.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Visual path is not a supported preview image: {path}")
    return resolved


def build_visual_catalog(cfg: dict) -> List[VisualCatalogItem]:
    """Build a validated catalog of generated visual outputs for the configured project."""

    manifest_items = _load_manifest_catalog(cfg)
    if manifest_items:
        return manifest_items
    preview_dir = resolve_project_preview_dir(cfg)
    output_dir = resolve_output_dir(cfg)
    if not preview_dir.exists():
        return []
    catalog: List[VisualCatalogItem] = []
    for path in sorted(preview_dir.iterdir(), key=lambda item: item.name.lower()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            valid = validate_visual_path(path, output_dir)
        except (FileNotFoundError, ValueError):
            continue
        catalog.append(_classify_preview_file(valid, output_dir))
    return catalog


def parse_visual_query(message: str) -> VisualQuery:
    """Parse visual intent into separate metric, variable, component, and visual-type signals."""

    text = normalize_visual_text(message)
    metric = _first_alias_match(text, METRIC_ALIASES)
    variable = _first_alias_match(text, _variable_aliases_with_generated_labels())
    visual_type = _first_alias_match(text, VISUAL_TYPE_ALIASES)
    if metric == "vr" and variable == "Yld_Mass_Dry_lb_ac" and visual_type != "interpolation_surface":
        variable = None
    component = _extract_component(text)
    if component and visual_type is None:
        visual_type = "component_map"
    if visual_type is None and re.search(r"\b(cluster|clustering|zone|zones)\b", text) and re.search(r"\b(result|results|output|outputs)\b", text):
        visual_type = "selected_map"
    wants_multiple = bool(re.search(r"\b(all|both|two)\b", text) or "compare the maps" in text)
    wants_best = "best" in text or "recommended" in text or "generated from" in text
    return VisualQuery(
        metric=metric,
        variable=variable,
        component=component,
        visual_type=visual_type,
        wants_multiple=wants_multiple,
        wants_best=wants_best,
        raw_text=text,
    )


def match_visual_catalog(query: VisualQuery, catalog: List[VisualCatalogItem]) -> List[VisualMatch]:
    """Rank catalog items against a parsed visual query using deterministic scoring."""

    matches: List[VisualMatch] = []
    query_terms = _meaningful_terms(query.raw_text)
    for item in catalog:
        score = 0.0
        reasons: List[str] = []
        item_aliases = [normalize_visual_text(alias) for alias in item.aliases]
        item_text = normalize_visual_text(" ".join([item.title, item.description, item.filename, *item.aliases]))

        if query.metric:
            if item.metric == query.metric:
                score += 40
                reasons.append("metric match")
            elif item.metric:
                score -= 40
                reasons.append("metric conflict")
        if query.variable:
            if item.variable == query.variable:
                score += 50
                reasons.append("variable match")
            elif item.variable:
                score -= 50
                reasons.append("variable conflict")
        if query.component:
            if item.component and normalize_visual_text(item.component) == normalize_visual_text(query.component):
                score += 50
                reasons.append("component match")
            elif item.component:
                score -= 50
                reasons.append("component conflict")
        if query.visual_type:
            if item.visual_type == query.visual_type:
                score += 35
                reasons.append("visual type match")
            else:
                score -= 50
                reasons.append("visual type conflict")
        if query.wants_best and item.visual_type == "selected_map":
            score += 15
            reasons.append("best selected map")
        for alias in item_aliases:
            if alias and alias in query.raw_text:
                score += 20
                reasons.append(f"alias '{alias}'")
                break
        overlap = len(query_terms.intersection(_meaningful_terms(item_text)))
        if overlap:
            score += overlap * 5
            reasons.append(f"{overlap} keyword overlap")
        if item.role and item.role in query.raw_text:
            score += 100
            reasons.append("role match")
        if score > 0:
            matches.append(VisualMatch(item=item, score=score, reasons=reasons))
    matches.sort(key=lambda match: (-match.score, _catalog_sort_key(match.item)))
    return matches


def resolve_visual_query(message: str, cfg: dict) -> VisualResolution:
    """Resolve a user visual request into images to render or choices to ask about."""

    catalog = build_visual_catalog(cfg)
    if not catalog:
        return VisualResolution(status="not_found", message="No generated preview images were found for this project.")
    query = parse_visual_query(message)
    if _is_best_figure_request(query):
        options = [item.id for item in sorted(catalog, key=_catalog_sort_key) if item.category in {"cluster_map", "metric_comparison"}][:5]
        if options:
            return VisualResolution(status="choices", option_ids=options, message=_choices_message(catalog, options))
    matches = match_visual_catalog(query, catalog)
    if not matches:
        if query.wants_best and ("figure" in query.raw_text or "image" in query.raw_text):
            options = [item.id for item in sorted(catalog, key=_catalog_sort_key) if item.category in {"cluster_map", "metric_comparison"}][:5]
            if options:
                return VisualResolution(status="choices", option_ids=options, message=_choices_message(catalog, options))
        categories = ", ".join(sorted({item.category for item in catalog}))
        return VisualResolution(status="not_found", message=f"I could not find a matching preview. Available visual categories: {categories}.")

    top = matches[0]
    plausible_threshold = max(20, top.score - 25) if top.score >= 20 else top.score
    plausible = [match for match in matches if match.score >= plausible_threshold]

    if query.wants_multiple:
        selected = _clear_multiple_selection(query, matches)
        if selected:
            return VisualResolution(
                status="resolved",
                selected_ids=[match.item.id for match in selected],
                message=_display_message([match.item for match in selected]),
            )

    if query.visual_type == "selected_map" and "clustering" in query.raw_text and "results" in query.raw_text and not query.metric:
        selected = [match for match in matches if match.item.category == "cluster_map"][:2]
        if selected:
            return VisualResolution(
                status="resolved",
                selected_ids=[match.item.id for match in selected],
                message=_display_message([match.item for match in selected]),
            )

    if _is_ambiguous_metric_image_request(query, plausible):
        options = _option_ids(plausible, limit=5)
        return VisualResolution(status="choices", option_ids=options, message=_choices_message(catalog, options))

    if _is_best_figure_ambiguous(query, plausible):
        options = _option_ids(plausible, limit=5)
        return VisualResolution(status="choices", option_ids=options, message=_choices_message(catalog, options))

    if _has_clear_single_resolution(query, top, plausible):
        return VisualResolution(status="resolved", selected_ids=[top.item.id], message=_display_message([top.item]))

    if top.score >= 75 and (len(plausible) == 1 or top.score - plausible[1].score >= 25):
        return VisualResolution(status="resolved", selected_ids=[top.item.id], message=_display_message([top.item]))

    options = _option_ids(plausible, limit=5)
    return VisualResolution(status="choices", option_ids=options, message=_choices_message(catalog, options))


def visual_items_by_ids(catalog: List[VisualCatalogItem], ids: List[str]) -> List[VisualCatalogItem]:
    """Return catalog items matching ids, preserving id order."""

    by_id = {item.id: item for item in catalog}
    return [by_id[item_id] for item_id in ids if item_id in by_id]


def resolve_pending_visual_selection(
    message: str,
    state,
    catalog: List[VisualCatalogItem],
) -> Optional[List[VisualCatalogItem]]:
    """Resolve a short follow-up against pending visual choice ids only."""

    pending_ids = list(getattr(state, "pending_visual_option_ids", []) or [])
    if not pending_ids:
        return None
    pending = visual_items_by_ids(catalog, pending_ids)
    if not pending:
        return None
    raw = message.strip()
    if raw in pending_ids:
        return visual_items_by_ids(catalog, [raw])
    text = normalize_visual_text(message)
    if text in {"show both", "both", "all", "all of them", "show all", "show them"}:
        return pending[:4]
    index = _parse_selection_index(text)
    if index is not None and 0 <= index < len(pending):
        return [pending[index]]
    query = parse_visual_query(message)
    matches = match_visual_catalog(query, pending)
    if not matches:
        return None
    top = matches[0]
    plausible = [match for match in matches if match.score >= max(20, top.score - 20)]
    if top.score >= 30 and (len(plausible) == 1 or top.score - plausible[1].score >= 10):
        return [top.item]
    if query.wants_multiple and plausible:
        return [match.item for match in plausible[:4]]
    return None


def resolve_visual_by_role(cfg: dict, role: str) -> Optional[Path]:
    """Resolve a semantic visual role to an existing preview image path."""

    for item in build_visual_catalog(cfg):
        if item.role == role:
            return Path(item.path)
    return None


def resolve_cluster_visuals(cfg: dict) -> Dict[str, Path]:
    """Return available cluster-map preview images keyed by semantic role."""

    return {item.role: Path(item.path) for item in build_visual_catalog(cfg) if item.category == "cluster_map" and item.role}


def resolve_comparison_visuals(cfg: dict) -> Dict[str, Path]:
    """Return available metric-comparison preview images keyed by semantic role."""

    return {item.role: Path(item.path) for item in build_visual_catalog(cfg) if item.category == "metric_comparison" and item.role}


def resolve_visuals_by_prefix(cfg: dict, prefix: str, role_prefix: str) -> Dict[str, Path]:
    """Return available preview images that follow a known prefix convention."""

    normalized_prefix = normalize_visual_text(prefix)
    return {
        f"{role_prefix}_{item.id}": Path(item.path)
        for item in build_visual_catalog(cfg)
        if normalize_visual_text(item.filename).startswith(normalized_prefix)
    }


def caption_for_role(role: Optional[str]) -> str:
    """Return a deterministic caption for a known visual role."""

    if not role:
        return "Generated project preview image."
    return VISUAL_CAPTIONS.get(role, "Generated project preview image.")


def visual_option_payload(item: VisualCatalogItem):
    """Return lightweight option fields usable by session-state models without importing them here."""

    return {
        "id": item.id,
        "title": item.title,
        "description": item.description,
        "role": item.role,
        "category": item.category,
        "visual_type": item.visual_type,
        "metric": item.metric,
        "variable": item.variable,
        "preview_path": item.path,
    }


def write_visual_manifest(cfg: dict) -> Optional[Path]:
    """Write a visual manifest from scanned preview outputs and return its path."""

    run_dir = resolve_project_raster_dir(cfg)
    preview_dir = resolve_project_preview_dir(cfg)
    if not preview_dir.exists():
        return None
    catalog = []
    output_dir = resolve_output_dir(cfg)
    for path in sorted(preview_dir.iterdir(), key=lambda item: item.name.lower()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            valid = validate_visual_path(path, output_dir)
        except (FileNotFoundError, ValueError):
            continue
        item = _classify_preview_file(valid, output_dir)
        catalog.append(_manifest_record(item))
    manifest_path = run_dir / "visual_manifest.json"
    manifest_path.write_text(json.dumps({"visuals": catalog}, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _load_manifest_catalog(cfg: dict) -> List[VisualCatalogItem]:
    manifest_path = resolve_project_raster_dir(cfg) / "visual_manifest.json"
    output_dir = resolve_output_dir(cfg)
    if not manifest_path.exists():
        return []
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    raw_items = data.get("visuals", data if isinstance(data, list) else [])
    if not isinstance(raw_items, list):
        return []
    catalog: List[VisualCatalogItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        path_text = raw.get("path") or raw.get("relative_path")
        if not path_text:
            continue
        candidate = Path(path_text)
        if not candidate.is_absolute():
            candidate = output_dir / candidate
        try:
            valid = validate_visual_path(candidate, output_dir)
        except (FileNotFoundError, ValueError):
            continue
        relative = valid.relative_to(output_dir).as_posix()
        try:
            catalog.append(
                VisualCatalogItem(
                    id=str(raw.get("id") or _id_from_filename(valid.name)),
                    path=relative,
                    filename=valid.name,
                    title=str(raw.get("title") or valid.stem.replace("_", " ").title()),
                    description=str(raw.get("description") or "Generated project preview image."),
                    category=str(raw.get("category") or "other_preview"),
                    visual_type=str(raw.get("visual_type") or "preview"),
                    role=raw.get("role"),
                    metric=raw.get("metric"),
                    variable=raw.get("variable"),
                    component=raw.get("component"),
                    aliases=list(raw.get("aliases") or []),
                )
            )
        except Exception:
            continue
    return catalog


def _classify_preview_file(path: Path, output_dir: Path) -> VisualCatalogItem:
    filename = path.name
    lower = filename.lower()
    relative = path.relative_to(output_dir).as_posix()
    if lower == "clusters_best.png":
        return VisualCatalogItem(
            id="clusters_best",
            path=relative,
            filename=filename,
            title="Best cluster map by outcome variance reduction",
            description="Cluster map selected using outcome variance reduction.",
            category="cluster_map",
            visual_type="selected_map",
            role="yield_variance_cluster_map",
            metric="vr",
            aliases=["yield variance map", "yield variance cluster map", "best yield variance result", "recommended map"],
        )
    if lower == "clusters_best_ch_score.png":
        return VisualCatalogItem(
            id="clusters_best_ch_score",
            path=relative,
            filename=filename,
            title="Best cluster map by Calinski-Harabasz score",
            description="Cluster map selected using the Calinski-Harabasz score.",
            category="cluster_map",
            visual_type="selected_map",
            role="ch_cluster_map",
            metric="ch_score",
            aliases=["ch map", "ch score map", "calinski harabasz map", "best ch score result"],
        )
    comparison = re.match(r"^comparison_(.+)\.(png|jpg|jpeg)$", filename, flags=re.IGNORECASE)
    if comparison:
        metric = _metric_from_filename_token(comparison.group(1))
        role = f"{metric}_comparison" if metric else None
        return VisualCatalogItem(
            id=_id_from_filename(filename),
            path=relative,
            filename=filename,
            title=f"{_metric_title(metric or comparison.group(1))} comparison figure",
            description=f"Comparison figure for {_metric_title(metric or comparison.group(1))}.",
            category="metric_comparison",
            visual_type="comparison_figure",
            role=role,
            metric=metric,
            aliases=_aliases_for_metric(metric) + ["comparison graph", "comparison image"],
        )
    component = re.match(r"^feature_(.+)\.(png|jpg|jpeg)$", filename, flags=re.IGNORECASE)
    if component:
        component_name = component.group(1)
        return VisualCatalogItem(
            id=_id_from_filename(filename),
            path=relative,
            filename=filename,
            title=f"{component_name.upper()} spatial component map",
            description=f"Spatial PCA/component preview for {component_name.upper()}.",
            category="component_map",
            visual_type="component_map",
            role=f"pca_component_{normalize_visual_text(component_name).replace(' ', '_')}",
            component=component_name.upper(),
            aliases=[component_name, component_name.upper(), component_name.lower(), component_name.replace("_", " ")],
        )
    kriging = re.match(r"^kriging_(.+)\.(png|jpg|jpeg)$", filename, flags=re.IGNORECASE)
    if kriging:
        variable = kriging.group(1)
        return VisualCatalogItem(
            id=_id_from_filename(filename),
            path=relative,
            filename=filename,
            title=f"{_variable_title(variable)} interpolation",
            description=f"Kriged spatial surface for {_variable_title(variable)}.",
            category="interpolation",
            visual_type="interpolation_surface",
            role=f"kriging_{normalize_visual_text(variable).replace(' ', '_')}",
            variable=variable,
            aliases=_aliases_for_variable(variable),
        )
    return VisualCatalogItem(
        id=_id_from_filename(filename),
        path=relative,
        filename=filename,
        title=path.stem.replace("_", " ").title(),
        description="Generated project preview image.",
        category="other_preview",
        visual_type="preview",
        aliases=[path.stem, path.stem.replace("_", " ")],
    )


def _first_alias_match(text: str, alias_map: Dict[str, List[str]]) -> Optional[str]:
    for key, aliases in alias_map.items():
        for alias in aliases:
            normalized = normalize_visual_text(alias)
            if re.search(rf"\b{re.escape(normalized)}\b", text):
                return key
    return None


def _variable_aliases_with_generated_labels() -> Dict[str, List[str]]:
    aliases: Dict[str, List[str]] = {}
    for variable, values in VARIABLE_ALIASES.items():
        generated = {variable, variable.replace("_", " "), variable.lower().replace("_", " ")}
        aliases[variable] = list(dict.fromkeys([*values, *generated]))
    return aliases


def _extract_component(text: str) -> Optional[str]:
    match = re.search(r"\b(?:pc|component|principal component)\s*([0-9]+)\b", text)
    if match:
        return f"PC{match.group(1)}"
    return None


def _meaningful_terms(text: str) -> set[str]:
    return {term for term in normalize_visual_text(text).split() if len(term) > 1 and term not in MEANINGLESS_TERMS}


def _catalog_sort_key(item: VisualCatalogItem) -> tuple[int, str]:
    category_order = {"cluster_map": 0, "metric_comparison": 1, "interpolation": 2, "component_map": 3, "other_preview": 4}
    metric_order = {"vr": 0, "ch_score": 1, "asc": 2, "anova_p": 3}
    return (category_order.get(item.category, 9), f"{metric_order.get(item.metric or '', 9)}_{item.filename.lower()}")


def _clear_multiple_selection(query: VisualQuery, matches: List[VisualMatch]) -> List[VisualMatch]:
    if query.visual_type:
        selected = [match for match in matches if match.item.visual_type == query.visual_type and match.score > 0]
    elif query.metric:
        selected = [match for match in matches if match.item.metric == query.metric and match.score > 0]
    else:
        selected = [match for match in matches if match.score > 0]
    return selected[:4]


def _is_ambiguous_metric_image_request(query: VisualQuery, plausible: List[VisualMatch]) -> bool:
    if not query.metric or query.visual_type is not None:
        return False
    if "image" not in query.raw_text and "figure" not in query.raw_text:
        return False
    types = {match.item.visual_type for match in plausible if match.item.metric == query.metric}
    return len(types.intersection({"selected_map", "comparison_figure"})) > 1


def _is_best_figure_ambiguous(query: VisualQuery, plausible: List[VisualMatch]) -> bool:
    return _is_best_figure_request(query) and len(plausible) > 1


def _is_best_figure_request(query: VisualQuery) -> bool:
    return query.wants_best and query.visual_type is None and ("figure" in query.raw_text or "image" in query.raw_text)


def _has_clear_single_resolution(query: VisualQuery, top: VisualMatch, plausible: List[VisualMatch]) -> bool:
    if query.metric and query.visual_type and top.item.metric == query.metric and top.item.visual_type == query.visual_type:
        return True
    if query.variable and query.visual_type == "interpolation_surface" and top.item.variable == query.variable:
        return True
    if query.component and top.item.component and normalize_visual_text(top.item.component) == normalize_visual_text(query.component):
        return True
    return len(plausible) == 1 and top.score >= 35


def _option_ids(matches: List[VisualMatch], limit: int) -> List[str]:
    ids: List[str] = []
    for match in matches:
        if match.item.id not in ids:
            ids.append(match.item.id)
        if len(ids) >= limit:
            break
    return ids


def _choices_message(catalog: List[VisualCatalogItem], option_ids: List[str]) -> str:
    items = visual_items_by_ids(catalog, option_ids)
    lines = ["I found more than one matching visual. Which one should I show?"]
    for index, item in enumerate(items, start=1):
        lines.append(f"{index}. {item.title}")
    return "\n".join(lines)


def _display_message(items: List[VisualCatalogItem]) -> str:
    if len(items) == 1:
        return items[0].description
    return "Here are the requested visual results: " + "; ".join(item.title for item in items) + "."


def _parse_selection_index(text: str) -> Optional[int]:
    word_map = {"first": 0, "the first one": 0, "option 1": 0, "1": 0, "second": 1, "the second one": 1, "option 2": 1, "2": 1, "third": 2, "option 3": 2, "3": 2, "fourth": 3, "option 4": 3, "4": 3}
    return word_map.get(text)


def _metric_from_filename_token(token: str) -> Optional[str]:
    normalized = normalize_visual_text(token)
    if normalized in {"vr"}:
        return "vr"
    if normalized in {"ch score", "ch"}:
        return "ch_score"
    if normalized == "asc":
        return "asc"
    if normalized in {"anova p", "anova"}:
        return "anova_p"
    return _first_alias_match(normalized, METRIC_ALIASES)


def _metric_title(metric: str) -> str:
    return {
        "vr": "outcome variance reduction",
        "ch_score": "Calinski-Harabasz score",
        "asc": "average silhouette coefficient",
        "anova_p": "outcome ANOVA p-value",
    }.get(metric, metric.replace("_", " "))


def _variable_title(variable: str) -> str:
    return variable.replace("_", " ")


def _aliases_for_metric(metric: Optional[str]) -> List[str]:
    if not metric:
        return []
    return METRIC_ALIASES.get(metric, [])


def _aliases_for_variable(variable: str) -> List[str]:
    known = VARIABLE_ALIASES.get(variable, [])
    generated = [variable, variable.replace("_", " "), variable.lower().replace("_", " ")]
    return list(dict.fromkeys([*known, *generated, "interpolation", "surface", "kriging"]))


def _id_from_filename(filename: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", Path(filename).stem.lower()).strip("_")


def _manifest_record(item: VisualCatalogItem) -> Dict[str, object]:
    return {
        "id": item.id,
        "filename": item.filename,
        "relative_path": item.path,
        "category": item.category,
        "visual_type": item.visual_type,
        "title": item.title,
        "description": item.description,
        "role": item.role,
        "metric": item.metric,
        "variable": item.variable,
        "component": item.component,
        "aliases": item.aliases,
    }
