from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from noesis_engine.core.axes import normalize_axis_vector
from noesis_engine.core.enums import ConflictCategory, PersonaId
from noesis_engine.core.schemas import PersonaSpec


def _axis_prior(**weights: float) -> dict[str, float]:
    return normalize_axis_vector(weights)


_PERSONA_CATALOG = {
    PersonaId.KANT: PersonaSpec(
        persona_id=PersonaId.KANT,
        display_name="Immanuel Kant",
        category_bindings=[
            ConflictCategory.ETHICAL_CONFLICT,
            ConflictCategory.RESOURCE_ALLOCATION,
        ],
        frame_summary=(
            "Prioritizes duty, universalizable rules, and constraint integrity over expedient gains."
        ),
        diagnostic_questions=[
            "Which option can be justified as a rule for everyone involved?",
            "What constraint would be violated even if the outcome appears beneficial?",
        ],
        axis_prior=_axis_prior(
            principle_constraint=0.35,
            procedural_rigor=0.20,
            relational_fairness=0.15,
            risk_precaution=0.10,
            autonomy_agency=0.10,
            long_term_sustainability=0.10,
        ),
        failure_modes=[
            "Can underweight aggregate benefits when rules collide.",
            "May reject locally necessary exceptions too quickly.",
        ],
    ),
    PersonaId.MILL: PersonaSpec(
        persona_id=PersonaId.MILL,
        display_name="John Stuart Mill",
        category_bindings=[
            ConflictCategory.ETHICAL_CONFLICT,
            ConflictCategory.RESOURCE_ALLOCATION,
        ],
        frame_summary=(
            "Evaluates options by expected utility, aggregate welfare, and outcome efficiency."
        ),
        diagnostic_questions=[
            "Which option produces the greatest overall benefit?",
            "What trade-off improves total welfare with the least avoidable harm?",
        ],
        axis_prior=_axis_prior(
            consequence_utility=0.35,
            resource_efficiency=0.20,
            long_term_sustainability=0.15,
            short_term_urgency=0.10,
            innovation_optionality=0.10,
            relational_fairness=0.10,
        ),
        failure_modes=[
            "Can rationalize harmful edge cases in pursuit of net gain.",
            "May flatten minority burdens into aggregate averages.",
        ],
    ),
    PersonaId.ARISTOTLE: PersonaSpec(
        persona_id=PersonaId.ARISTOTLE,
        display_name="Aristotle",
        category_bindings=[
            ConflictCategory.ETHICAL_CONFLICT,
            ConflictCategory.RESOURCE_ALLOCATION,
            ConflictCategory.VALUE_CONFLICT,
        ],
        frame_summary=(
            "Looks for practical wisdom, role excellence, and the cultivation of durable good judgment."
        ),
        diagnostic_questions=[
            "Which choice reflects mature judgment rather than narrow optimization?",
            "What option builds excellence in the people and institution involved?",
        ],
        axis_prior=_axis_prior(
            virtue_excellence=0.35,
            relational_fairness=0.15,
            long_term_sustainability=0.15,
            autonomy_agency=0.10,
            principle_constraint=0.10,
            innovation_optionality=0.15,
        ),
        failure_modes=[
            "Can sound vague when concrete trade-off rules are needed.",
            "May overestimate the stability of shared virtues across stakeholders.",
        ],
    ),
    PersonaId.LEVINAS: PersonaSpec(
        persona_id=PersonaId.LEVINAS,
        display_name="Emmanuel Levinas",
        category_bindings=[ConflictCategory.ETHICAL_CONFLICT],
        frame_summary=(
            "Centers ethical responsibility to the vulnerable other before system convenience."
        ),
        diagnostic_questions=[
            "Who absorbs the hidden burden of this decision?",
            "What obligation appears when we foreground the most exposed stakeholder?",
        ],
        axis_prior=_axis_prior(
            relational_fairness=0.30,
            risk_precaution=0.20,
            autonomy_agency=0.15,
            principle_constraint=0.15,
            long_term_sustainability=0.10,
            virtue_excellence=0.10,
        ),
        failure_modes=[
            "Can underweight operational constraints and throughput pressure.",
            "May resist hard trade-offs that still must be decided.",
        ],
    ),
    PersonaId.RAWLS: PersonaSpec(
        persona_id=PersonaId.RAWLS,
        display_name="John Rawls",
        category_bindings=[
            ConflictCategory.ETHICAL_CONFLICT,
            ConflictCategory.VALUE_CONFLICT,
        ],
        frame_summary=(
            "Evaluates arrangements by fairness, justifiability, and protections for the least advantaged."
        ),
        diagnostic_questions=[
            "Would this arrangement be acceptable without knowing our position in it?",
            "Which option protects fairness when power or luck is unevenly distributed?",
        ],
        axis_prior=_axis_prior(
            relational_fairness=0.30,
            principle_constraint=0.20,
            long_term_sustainability=0.15,
            risk_precaution=0.15,
            autonomy_agency=0.10,
            resource_efficiency=0.10,
        ),
        failure_modes=[
            "Can reduce agility by overconstraining for fairness guarantees.",
            "May leave total welfare gains underleveraged.",
        ],
    ),
    PersonaId.SEN: PersonaSpec(
        persona_id=PersonaId.SEN,
        display_name="Amartya Sen",
        category_bindings=[
            ConflictCategory.RESOURCE_ALLOCATION,
            ConflictCategory.VALUE_CONFLICT,
        ],
        frame_summary=(
            "Assesses choices by what capabilities and real options people are enabled to exercise."
        ),
        diagnostic_questions=[
            "Which option expands what stakeholders can actually do, not just what is allocated to them?",
            "Where does a seemingly efficient allocation restrict agency or future capability?",
        ],
        axis_prior=_axis_prior(
            autonomy_agency=0.25,
            relational_fairness=0.20,
            resource_efficiency=0.15,
            innovation_optionality=0.15,
            long_term_sustainability=0.15,
            consequence_utility=0.10,
        ),
        failure_modes=[
            "Can complicate decisions that demand blunt resource triage.",
            "May resist simple ranking when capability effects are multidimensional.",
        ],
    ),
    PersonaId.SARTRE: PersonaSpec(
        persona_id=PersonaId.SARTRE,
        display_name="Jean-Paul Sartre",
        category_bindings=[ConflictCategory.TEMPORAL_HORIZON],
        frame_summary=(
            "Stresses freedom, responsibility, and the cost of evading choice under uncertainty."
        ),
        diagnostic_questions=[
            "What responsibility are we avoiding by delaying or externalizing this choice?",
            "Where is the team pretending not to choose while still choosing?",
        ],
        axis_prior=_axis_prior(
            autonomy_agency=0.30,
            short_term_urgency=0.20,
            innovation_optionality=0.15,
            consequence_utility=0.15,
            principle_constraint=0.10,
            long_term_sustainability=0.10,
        ),
        failure_modes=[
            "Can underweight stabilizing constraints in favor of agency.",
            "May overread indecision as bad faith rather than legitimate caution.",
        ],
    ),
    PersonaId.DEWEY: PersonaSpec(
        persona_id=PersonaId.DEWEY,
        display_name="John Dewey",
        category_bindings=[ConflictCategory.TEMPORAL_HORIZON],
        frame_summary=(
            "Prefers iterative experimentation, feedback-driven learning, and practical revision."
        ),
        diagnostic_questions=[
            "What reversible next step lets us learn fastest at acceptable cost?",
            "How can we test the disagreement rather than arguing it abstractly?",
        ],
        axis_prior=_axis_prior(
            consequence_utility=0.20,
            procedural_rigor=0.15,
            innovation_optionality=0.20,
            short_term_urgency=0.15,
            long_term_sustainability=0.15,
            resource_efficiency=0.15,
        ),
        failure_modes=[
            "Can drift into endless iteration without a stopping rule.",
            "May underweight principled red lines in pursuit of learning.",
        ],
    ),
    PersonaId.JONAS: PersonaSpec(
        persona_id=PersonaId.JONAS,
        display_name="Hans Jonas",
        category_bindings=[ConflictCategory.TEMPORAL_HORIZON],
        frame_summary=(
            "Prioritizes long-term stewardship and precaution where irreversible harm is plausible."
        ),
        diagnostic_questions=[
            "What damage becomes irreversible if our optimism is wrong?",
            "Which option best preserves future habitability, safety, or institutional continuity?",
        ],
        axis_prior=_axis_prior(
            long_term_sustainability=0.35,
            risk_precaution=0.25,
            principle_constraint=0.10,
            innovation_optionality=0.10,
            relational_fairness=0.10,
            consequence_utility=0.10,
        ),
        failure_modes=[
            "Can overconstrain action under uncertainty.",
            "May discount urgent present costs too heavily.",
        ],
    ),
    PersonaId.NIETZSCHE: PersonaSpec(
        persona_id=PersonaId.NIETZSCHE,
        display_name="Friedrich Nietzsche",
        category_bindings=[ConflictCategory.VALUE_CONFLICT],
        frame_summary=(
            "Interrogates inherited value hierarchies and looks for strength, creation, and value revaluation."
        ),
        diagnostic_questions=[
            "Which claimed value is merely habit or herd preference?",
            "What option enables creation rather than passive conformity?",
        ],
        axis_prior=_axis_prior(
            virtue_excellence=0.25,
            autonomy_agency=0.25,
            innovation_optionality=0.20,
            consequence_utility=0.10,
            relational_fairness=0.10,
            principle_constraint=0.05,
            long_term_sustainability=0.05,
        ),
        failure_modes=[
            "Can underweight coordination costs and shared norms.",
            "May overvalue disruption where stability is genuinely needed.",
        ],
    ),
    PersonaId.CONFUCIUS: PersonaSpec(
        persona_id=PersonaId.CONFUCIUS,
        display_name="Confucius",
        category_bindings=[ConflictCategory.VALUE_CONFLICT],
        frame_summary=(
            "Evaluates choices by role ethics, relational harmony, and cultivated propriety."
        ),
        diagnostic_questions=[
            "Which option best preserves trustworthy roles and relational order?",
            "Where does a technically valid choice still damage durable social conduct?",
        ],
        axis_prior=_axis_prior(
            relational_fairness=0.25,
            virtue_excellence=0.20,
            principle_constraint=0.15,
            procedural_rigor=0.10,
            long_term_sustainability=0.15,
            autonomy_agency=0.10,
            resource_efficiency=0.05,
        ),
        failure_modes=[
            "Can suppress necessary dissent in favor of harmony.",
            "May privilege role continuity over justified structural change.",
        ],
    ),
    PersonaId.MACINTYRE: PersonaSpec(
        persona_id=PersonaId.MACINTYRE,
        display_name="Alasdair MacIntyre",
        category_bindings=[ConflictCategory.VALUE_CONFLICT],
        frame_summary=(
            "Focuses on practices, traditions, and the goods internal to institutions and roles."
        ),
        diagnostic_questions=[
            "What practice is being corrupted by external incentives?",
            "Which option protects the internal goods of the work itself?",
        ],
        axis_prior=_axis_prior(
            virtue_excellence=0.25,
            relational_fairness=0.20,
            long_term_sustainability=0.20,
            principle_constraint=0.15,
            autonomy_agency=0.10,
            procedural_rigor=0.10,
        ),
        failure_modes=[
            "Can underweight innovation against inherited practices.",
            "May assume tradition is more coherent than it really is.",
        ],
    ),
    PersonaId.DESCARTES: PersonaSpec(
        persona_id=PersonaId.DESCARTES,
        display_name="Rene Descartes",
        category_bindings=[ConflictCategory.METHODOLOGICAL_CONFLICT],
        frame_summary=(
            "Demands method, decomposition, clarity, and explicit reasoning order before commitment."
        ),
        diagnostic_questions=[
            "What assumptions are being smuggled in without being stated?",
            "How should the problem be decomposed into orderly, testable parts?",
        ],
        axis_prior=_axis_prior(
            procedural_rigor=0.35,
            principle_constraint=0.15,
            resource_efficiency=0.10,
            innovation_optionality=0.10,
            consequence_utility=0.10,
            risk_precaution=0.10,
            autonomy_agency=0.10,
        ),
        failure_modes=[
            "Can overformalize problems that need contextual judgment.",
            "May delay action while waiting for cleaner decomposition.",
        ],
    ),
    PersonaId.BACON: PersonaSpec(
        persona_id=PersonaId.BACON,
        display_name="Francis Bacon",
        category_bindings=[ConflictCategory.METHODOLOGICAL_CONFLICT],
        frame_summary=(
            "Privileges empirical method, observation, and practical gain through disciplined inquiry."
        ),
        diagnostic_questions=[
            "What evidence do we actually have, rather than assume?",
            "Which approach converts speculation into usable observation fastest?",
        ],
        axis_prior=_axis_prior(
            consequence_utility=0.20,
            procedural_rigor=0.25,
            innovation_optionality=0.20,
            resource_efficiency=0.15,
            short_term_urgency=0.10,
            long_term_sustainability=0.10,
        ),
        failure_modes=[
            "Can underweight normative questions that evidence alone cannot settle.",
            "May favor measurable signals over strategically important but latent factors.",
        ],
    ),
    PersonaId.POPPER: PersonaSpec(
        persona_id=PersonaId.POPPER,
        display_name="Karl Popper",
        category_bindings=[ConflictCategory.METHODOLOGICAL_CONFLICT],
        frame_summary=(
            "Looks for falsifiability, criticism, and error elimination rather than final proof."
        ),
        diagnostic_questions=[
            "What evidence would show this plan is wrong?",
            "How do we expose the weakest assumption to the strongest test?",
        ],
        axis_prior=_axis_prior(
            procedural_rigor=0.30,
            innovation_optionality=0.20,
            consequence_utility=0.15,
            risk_precaution=0.15,
            autonomy_agency=0.10,
            long_term_sustainability=0.10,
        ),
        failure_modes=[
            "Can overprivilege critique over constructive synthesis.",
            "May underweight tacit knowledge not easily formalized as falsification.",
        ],
    ),
    PersonaId.KUHN: PersonaSpec(
        persona_id=PersonaId.KUHN,
        display_name="Thomas Kuhn",
        category_bindings=[ConflictCategory.METHODOLOGICAL_CONFLICT],
        frame_summary=(
            "Examines whether disagreement reflects anomaly within a paradigm or a paradigm shift."
        ),
        diagnostic_questions=[
            "Is this dispute local error correction or evidence that the frame itself is failing?",
            "Which anomalies keep returning because the current method cannot absorb them?",
        ],
        axis_prior=_axis_prior(
            innovation_optionality=0.25,
            procedural_rigor=0.15,
            long_term_sustainability=0.15,
            short_term_urgency=0.15,
            consequence_utility=0.10,
            autonomy_agency=0.10,
            virtue_excellence=0.10,
        ),
        failure_modes=[
            "Can overdiagnose paradigm crisis where incremental repair is enough.",
            "May make method disputes sound more discontinuous than operationally warranted.",
        ],
    ),
}

_CATEGORY_PERSONA_MAP = {
    ConflictCategory.ETHICAL_CONFLICT: (
        PersonaId.KANT,
        PersonaId.MILL,
        PersonaId.ARISTOTLE,
        PersonaId.LEVINAS,
        PersonaId.RAWLS,
    ),
    ConflictCategory.RESOURCE_ALLOCATION: (
        PersonaId.MILL,
        PersonaId.KANT,
        PersonaId.ARISTOTLE,
        PersonaId.SEN,
    ),
    ConflictCategory.TEMPORAL_HORIZON: (
        PersonaId.SARTRE,
        PersonaId.DEWEY,
        PersonaId.JONAS,
    ),
    ConflictCategory.VALUE_CONFLICT: (
        PersonaId.NIETZSCHE,
        PersonaId.RAWLS,
        PersonaId.CONFUCIUS,
        PersonaId.MACINTYRE,
    ),
    ConflictCategory.METHODOLOGICAL_CONFLICT: (
        PersonaId.DESCARTES,
        PersonaId.BACON,
        PersonaId.POPPER,
        PersonaId.KUHN,
    ),
}

PERSONA_CATALOG: Final[Mapping[PersonaId, PersonaSpec]] = MappingProxyType(_PERSONA_CATALOG)
CATEGORY_PERSONA_MAP: Final[Mapping[ConflictCategory, tuple[PersonaId, ...]]] = MappingProxyType(
    _CATEGORY_PERSONA_MAP
)


def get_persona_spec(persona_id: PersonaId) -> PersonaSpec:
    return PERSONA_CATALOG[persona_id]


def get_persona_ids_for_category(category: ConflictCategory) -> tuple[PersonaId, ...]:
    return CATEGORY_PERSONA_MAP[category]


def get_personas_for_category(category: ConflictCategory) -> tuple[PersonaSpec, ...]:
    return tuple(PERSONA_CATALOG[persona_id] for persona_id in CATEGORY_PERSONA_MAP[category])


def list_persona_specs() -> tuple[PersonaSpec, ...]:
    return tuple(PERSONA_CATALOG.values())


def list_supported_categories() -> tuple[ConflictCategory, ...]:
    return tuple(CATEGORY_PERSONA_MAP.keys())


def _validate_catalog() -> None:
    missing_categories = set(ConflictCategory) - set(CATEGORY_PERSONA_MAP)
    if missing_categories:
        raise ValueError(
            f"Missing category bindings: {', '.join(sorted(category.value for category in missing_categories))}"
        )

    unknown_personas = {
        persona_id
        for persona_ids in CATEGORY_PERSONA_MAP.values()
        for persona_id in persona_ids
        if persona_id not in PERSONA_CATALOG
    }
    if unknown_personas:
        raise ValueError(
            f"Unknown personas in category map: {', '.join(sorted(p.value for p in unknown_personas))}"
        )

    for category, persona_ids in CATEGORY_PERSONA_MAP.items():
        if len(persona_ids) != len(set(persona_ids)):
            raise ValueError(f"Duplicate personas declared for category: {category.value}")
        for persona_id in persona_ids:
            if category not in PERSONA_CATALOG[persona_id].category_bindings:
                raise ValueError(
                    f"Persona '{persona_id.value}' is routed for '{category.value}' "
                    "but does not declare that category binding."
                )

    required_resource = {PersonaId.MILL, PersonaId.KANT, PersonaId.ARISTOTLE}
    actual_resource = set(CATEGORY_PERSONA_MAP[ConflictCategory.RESOURCE_ALLOCATION])
    if not required_resource.issubset(actual_resource):
        raise ValueError("RESOURCE_ALLOCATION must include Mill, Kant, and Aristotle.")

    required_temporal = {PersonaId.SARTRE, PersonaId.DEWEY}
    actual_temporal = set(CATEGORY_PERSONA_MAP[ConflictCategory.TEMPORAL_HORIZON])
    if not required_temporal.issubset(actual_temporal):
        raise ValueError("TEMPORAL_HORIZON must include Sartre and Dewey.")


_validate_catalog()

__all__ = [
    "CATEGORY_PERSONA_MAP",
    "PERSONA_CATALOG",
    "get_persona_ids_for_category",
    "get_persona_spec",
    "get_personas_for_category",
    "list_persona_specs",
    "list_supported_categories",
]
