from __future__ import annotations

from autocapture.ui.api_supervisor import ApiSupervisor


def test_api_supervisor_backoff_and_restart() -> None:
    now = 0.0

    def time_source() -> float:
        return now

    restart_calls: list[float] = []

    def restart() -> None:
        restart_calls.append(now)

    def health_check() -> bool:
        return False

    supervisor = ApiSupervisor(
        health_check,
        restart,
        backoff_base_s=2.0,
        backoff_max_s=10.0,
        time_source=time_source,
    )

    state = supervisor.tick()
    assert state.status == "restarting"
    assert restart_calls == [0.0]

    state = supervisor.tick()
    assert state.status == "backoff"
    assert restart_calls == [0.0]

    now = 3.0
    state = supervisor.tick()
    assert state.status == "restarting"
    assert len(restart_calls) == 2
