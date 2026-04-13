from __future__ import annotations

from noesis_engine.core.enums import ConflictCategory
from noesis_engine.core.persona_catalog import get_personas_for_category
from noesis_engine.core.schemas import CategoryResult, PersonaSpec


class PersonaRouterService:
    def route(self, category: ConflictCategory) -> tuple[PersonaSpec, ...]:
        return get_personas_for_category(category)

    def route_from_result(self, category_result: CategoryResult) -> tuple[PersonaSpec, ...]:
        return self.route(category_result.primary_category)


__all__ = ["PersonaRouterService"]
