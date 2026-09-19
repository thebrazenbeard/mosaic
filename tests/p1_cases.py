from runtime.handoff import ContinuityCase, StateEvent


def _case(case_id, family, key, values):
    events = tuple(
        StateEvent(
            event_id=f"{case_id}-e{index}",
            family=family,
            key=key,
            value=value,
            revision=index,
        )
        for index, value in enumerate(values, start=1)
    )
    return ContinuityCase(
        case_id=case_id,
        family=family,
        events=events,
        query_key=key,
        expected_value=values[-1],
    )


CASES = (
    _case("referent", "REFERENT_PRESERVATION", "referent", ("object-17",)),
    _case("correction", "CORRECTION", "fact", ("blue", "green")),
    _case("supersession", "SUPERSESSION_CURRENTNESS", "policy", ("draft-v1", "current-v2")),
    _case("decision", "DECISION_FINALITY", "decision", ("tentative", "final-approved")),
    _case("authority", "AUTHORITY_CHANGE", "writer", ("worker-a", "worker-b")),
    _case("preference", "USER_PREFERENCE", "format", ("verbose", "compact")),
    _case("project", "PROJECT_STATUS", "status", ("in-progress", "blocked-on-review")),
    _case("artifact", "ARTIFACT_HANDOFF", "artifact", ("artifact-sha256-abc",)),
)
