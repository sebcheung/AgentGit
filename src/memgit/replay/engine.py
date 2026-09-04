"""The replay / ablation engine: causal attribution for a memory.

Takes a memory state, removes one belief, and reruns the same query against
both the original and the ablated state — a controlled experiment, not a
turn. Neither run ever calls :meth:`~memgit.core.repository.Repository.commit`:
an ablated state is, by ``MemoryState``'s own contract, not *the* state at
any real commit ("any derived state drops its commit provenance"), so
persisting one back into history would fabricate a belief the agent never
actually held.

Reuses :func:`memgit.agent.runtime.run_tool_loop` — the same tool-calling
loop a real turn runs — so a replay's ``remember``/``forget`` calls are
decoded and merged exactly as they would be live; only the commit step at
the end of :meth:`~memgit.agent.runtime.MemoryAgent.turn` is skipped.

See PLAN.md's "Honest caveats": a changed reply is evidence, not proof, that
the ablated fact caused it — removing one fact can have downstream effects
on other facts that reference it, and the model's phrasing can drift for
reasons unrelated to the ablated belief.

**Retrieval mode pins the retrieved set — it never re-retrieves per side.**
Naively calling :meth:`~memgit.retrieval.rank.Retriever.retrieve` separately
for the baseline and the ablated state would break the one-variable
property above: removing the ablated fact frees a slot in the top-k, a
different fact backfills it, and the two prompts then differ in *two*
facts instead of one — the control group changed, which is a defect in the
experiment, not the honest caveat this module already discloses.
:func:`ablate_and_replay` therefore retrieves exactly once, against the
baseline state, and the ablated side reuses that same retrieved set minus
the ablated fact, with no backfill. The residual bias — the pinned set was
ranked *with* the ablated fact present, so it may have displaced something
else from the top-k — is the honest trade for holding the stimulus
constant; see PLAN.md's "Honest caveats" for it stated plainly rather than
hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from memgit.agent.client import LLMClient, default_client
from memgit.agent.prompt import build_system_prompt
from memgit.agent.runtime import ToolCallRecord, run_tool_loop
from memgit.core.cardinality import CardinalityMap
from memgit.core.fact import FactKey
from memgit.core.repository import Repository
from memgit.core.state import MemoryState
from memgit.retrieval.rank import RetrievalResult, Retriever

__all__ = ["ReplayOutcome", "AblationResult", "replay_query", "ablate_and_replay"]

_DEFAULT_FULL_BELOW = 64


@dataclass(frozen=True, slots=True)
class ReplayOutcome:
    """What one isolated replay of a query against a fixed state produced."""

    reply: str
    stop_reason: str
    tool_calls: tuple[ToolCallRecord, ...]
    state: MemoryState


@dataclass(frozen=True, slots=True)
class AblationResult:
    """The same query, run against a state and against that state minus one belief.

    Attributes:
        key: The ``(subject, predicate)`` ablated.
        fact_hash: If only one value at ``key`` was removed (the
            multi-valued case), its hash; ``None`` when every value at
            ``key`` was removed.
        retrieved: The pinned retrieval result both sides' prompts were
            built from (baseline injects it as-is; the ablated side injects
            it minus the ablated fact, with no backfill) — ``None`` when
            this replay ran in full-injection mode. Provenance, mirroring
            why a diff embeds the cardinality map it used.
    """

    query: str
    key: FactKey
    fact_hash: str | None
    baseline: ReplayOutcome
    ablated: ReplayOutcome
    retrieved: RetrievalResult | None = None

    @property
    def changed(self) -> bool:
        """Whether removing the belief changed the model's reply at all.

        A cheap, honest signal — literal string inequality, not semantic
        equivalence. See the module docstring's caveat before treating a
        change as proof of causation.
        """
        return self.baseline.reply != self.ablated.reply


def replay_query(
    state: MemoryState,
    query: str,
    cardinality: CardinalityMap,
    *,
    client: LLMClient | None = None,
    model: str = "claude-opus-5",
    head_label: str | None = None,
    max_tool_rounds: int = 8,
    source: str = "replay",
    retrieved: RetrievalResult | None = None,
    retriever: Retriever | None = None,
    as_of: datetime | None = None,
) -> ReplayOutcome:
    """Run one isolated turn of ``query`` against ``state`` — never committed.

    A fresh, one-message conversation on every call: no transcript carried
    in from anywhere else, so two calls with the same ``query`` differ only
    in the ``state`` (and, under retrieval, the pinned ``retrieved`` set)
    passed in — the controlled half of the experiment :class:`AblationResult`
    reports on.

    Args:
        head_label: How the system prompt should describe where ``state``
            came from (e.g. ``"HEAD (a1b2c3d4)"``). Purely descriptive — it
            does not affect resolution, since a bare ``MemoryState`` may not
            correspond to any commit at all (an ablated one never does).
        retrieved: A pre-computed retrieval result to inject instead of the
            full state — see the module docstring for why this is computed
            once by the caller (:func:`ablate_and_replay`) rather than by
            this function re-retrieving per side.
        retriever: If given, offers the ``recall`` tool for this call,
            searching ``state`` live (recall is exploratory and read-only,
            so it is not part of the pinned-set experiment). Independent of
            ``retrieved`` — a caller may inject a pinned subset while still
            allowing live ``recall`` against the same state.
        as_of: Required when ``retriever`` is given; see
            :func:`~memgit.agent.runtime.run_tool_loop`.
    """
    resolved_client = client if client is not None else default_client(model=model)
    system = build_system_prompt(state, cardinality, head=head_label, retrieved=retrieved)
    messages: list[dict] = [{"role": "user", "content": query}]

    response, working, tool_calls = run_tool_loop(
        resolved_client,
        system,
        messages,
        state,
        cardinality,
        source=source,
        max_tool_rounds=max_tool_rounds,
        retriever=retriever,
        recall_state=state,
        as_of=as_of,
    )
    reply = "".join(block.text for block in response.content if block.type == "text")
    return ReplayOutcome(
        reply=reply, stop_reason=response.stop_reason, tool_calls=tuple(tool_calls), state=working
    )


def ablate_and_replay(
    repo: Repository,
    revision: str,
    key: FactKey,
    query: str,
    *,
    client: LLMClient | None = None,
    model: str = "claude-opus-5",
    max_tool_rounds: int = 8,
    fact_hash: str | None = None,
    retrieve: bool | None = None,
    k: int = 8,
    as_of: datetime | None = None,
) -> AblationResult:
    """Compare replaying ``query`` at ``revision`` with and without ``key``.

    Both replays share one ``client``, so a real :class:`AnthropicClient`
    is constructed only once per call — irrelevant to correctness (each
    replay is already a fresh conversation) but the honest choice for an
    experiment meant to hold everything but the ablated belief constant.

    Args:
        revision: The memory state to replay from — resolved once, so both
            the baseline and the ablated run compare against the exact same
            point in history even if a concurrent process moves ``HEAD``.
        key: The ``(subject, predicate)`` to ablate.
        fact_hash: If given, ablate only this one value at ``key`` (the
            multi-valued case, :meth:`MemoryState.without_fact`) instead of
            every value there (:meth:`MemoryState.without`).
        retrieve: Force retrieval mode on or off. ``None`` (the default)
            decides the same way :meth:`~memgit.agent.runtime.MemoryAgent.turn`
            does — by fact count against ``config["retrieval"]["full_below"]``
            — so a replay mirrors what a live turn would actually show.
        k: How many facts the pinned retrieval keeps.
        as_of: The moment retrieval ranks against. Defaults to now;
            explicit for reproducible replays.

    Raises:
        RevisionNotFoundError: ``revision`` does not resolve.
    """
    resolved_hash = repo.resolve(revision)
    state = repo.state(resolved_hash)
    cardinality = repo.cardinality()
    resolved_client = client if client is not None else default_client(model=model)
    label = f"{revision} ({resolved_hash[:8]})"
    moment = as_of if as_of is not None else datetime.now(timezone.utc)

    full_below = repo.config().get("retrieval", {}).get("full_below", _DEFAULT_FULL_BELOW)
    retrieval_mode = retrieve if retrieve is not None else len(state) >= full_below
    retriever = repo.retriever() if retrieval_mode else None

    # Retrieved exactly once, against the baseline state -- see the module
    # docstring for why the ablated side reuses this set instead of
    # re-retrieving, which would silently change more than one variable.
    baseline_retrieved = (
        retriever.retrieve(state, query, k=k, as_of=moment) if retriever is not None else None
    )

    baseline = replay_query(
        state,
        query,
        cardinality,
        client=resolved_client,
        model=model,
        head_label=label,
        max_tool_rounds=max_tool_rounds,
        source=f"replay:baseline/{resolved_hash[:8]}",
        retrieved=baseline_retrieved,
        retriever=retriever,
        as_of=moment,
    )

    ablated_state = state.without_fact(fact_hash) if fact_hash is not None else state.without(key)

    ablated_retrieved = None
    if baseline_retrieved is not None:
        if fact_hash is not None:
            kept = tuple(r for r in baseline_retrieved.facts if r.fact.hash != fact_hash)
        else:
            kept = tuple(r for r in baseline_retrieved.facts if r.fact.key != key)
        ablated_retrieved = RetrievalResult(
            query=baseline_retrieved.query,
            facts=kept,
            commit=ablated_state.commit,
            candidates=len(ablated_state.facts),
            embedder=baseline_retrieved.embedder,
            as_of=baseline_retrieved.as_of,
        )

    ablated = replay_query(
        ablated_state,
        query,
        cardinality,
        client=resolved_client,
        model=model,
        head_label=f"{label}, without {key[0]} {key[1]}",
        max_tool_rounds=max_tool_rounds,
        source=f"replay:ablated/{resolved_hash[:8]}",
        retrieved=ablated_retrieved,
        retriever=retriever,
        as_of=moment,
    )

    return AblationResult(
        query=query, key=key, fact_hash=fact_hash, baseline=baseline, ablated=ablated,
        retrieved=baseline_retrieved,
    )
