from noesis_engine.core.enums import ConflictCategory, PersonaId
from noesis_engine.services.persona_router import PersonaRouterService


def test_resource_allocation_route_is_deterministic() -> None:
    router = PersonaRouterService()
    first = tuple(persona.persona_id for persona in router.route(ConflictCategory.RESOURCE_ALLOCATION))
    second = tuple(persona.persona_id for persona in router.route(ConflictCategory.RESOURCE_ALLOCATION))

    assert first == second
    assert {PersonaId.MILL, PersonaId.KANT, PersonaId.ARISTOTLE}.issubset(set(first))


def test_temporal_horizon_route_contains_required_personas() -> None:
    router = PersonaRouterService()
    persona_ids = tuple(persona.persona_id for persona in router.route(ConflictCategory.TEMPORAL_HORIZON))

    assert PersonaId.SARTRE in persona_ids
    assert PersonaId.DEWEY in persona_ids
