from scripts import launch_training_study as supervisor


def state(**changes):
    value=dict(status='running',heartbeat_unix=1000,supervisor_pid=42,supervisor_birth='original-birth')
    value.update(changes)
    return value


def test_recent_heartbeat_does_not_hide_dead_supervisor(monkeypatch):
    monkeypatch.setattr(supervisor,'process_identity',lambda pid:None)
    assert supervisor.effective_status(state(),now=1001)=='supervisor_not_alive'


def test_reused_pid_is_not_original_supervisor(monkeypatch):
    monkeypatch.setattr(supervisor,'process_identity',lambda pid:'different-birth')
    assert supervisor.effective_status(state(),now=1001)=='supervisor_not_alive'


def test_stale_heartbeat_does_not_count_as_running_even_with_live_pid(monkeypatch):
    monkeypatch.setattr(supervisor,'process_identity',lambda pid:'original-birth')
    assert supervisor.effective_status(state(),now=1091)=='stale_heartbeat'


def test_live_identity_and_recent_heartbeat_count_as_running(monkeypatch):
    monkeypatch.setattr(supervisor,'process_identity',lambda pid:'original-birth')
    assert supervisor.effective_status(state(),now=1001)=='running'


def test_terminal_execution_error_is_not_relabeled_running(monkeypatch):
    def unexpected(pid):raise AssertionError('Terminal status does not query an unrelated PID')
    monkeypatch.setattr(supervisor,'process_identity',unexpected)
    assert supervisor.effective_status(state(status='execution_error'),now=1001)=='execution_error'
