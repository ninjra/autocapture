from autocapture.worker.agent_worker import AgentJobWorker


def test_connection_refused_detection_by_message() -> None:
    exc = ConnectionError("Connection refused")
    assert AgentJobWorker._is_connection_refused(exc) is True


def test_connection_refused_detection_false() -> None:
    exc = RuntimeError("Other failure")
    assert AgentJobWorker._is_connection_refused(exc) is False
