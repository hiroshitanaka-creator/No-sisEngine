from noesis_engine.core.enums import ConflictCategory, PersonaId
from noesis_engine.core.persona_catalog import (
    get_persona_ids_for_category,
    get_persona_spec,
    get_personas_for_category,
    list_persona_specs,
    list_supported_categories,
)
from noesis_engine.core.schemas import PersonaSpec


def test_get_persona_spec_returns_spec() -> None:
    spec = get_persona_spec(PersonaId.KANT)
    assert isinstance(spec, PersonaSpec)
    assert spec.persona_id == PersonaId.KANT


def test_get_persona_ids_for_category_returns_tuple() -> None:
    ids = get_persona_ids_for_category(ConflictCategory.RESOURCE_ALLOCATION)
    assert isinstance(ids, tuple)
    assert PersonaId.KANT in ids
    assert PersonaId.MILL in ids
    assert PersonaId.ARISTOTLE in ids


def test_get_personas_for_category_returns_specs() -> None:
    specs = get_personas_for_category(ConflictCategory.ETHICAL_CONFLICT)
    assert all(isinstance(s, PersonaSpec) for s in specs)
    assert len(specs) >= 1


def test_list_persona_specs_returns_all_16() -> None:
    specs = list_persona_specs()
    assert len(specs) == len(PersonaId)


def test_list_supported_categories_returns_all_categories() -> None:
    categories = list_supported_categories()
    assert set(categories) == set(ConflictCategory)
