from src.environment.mixed_logistics import make_mixed_scenario
from scripts.monitor_drift import order_features


def test_order_features_reflect_arrival_interval():
    for interval in (20, 7):
        _, manifest = make_mixed_scenario(520000, interval, 600, 5, 'drift_monitor')
        rows = order_features(manifest)
        assert len(rows) == len(manifest['events'])
        assert {r['inter_arrival'] for r in rows} == {0, interval}
        assert sum(r['outbound'] for r in rows) == len(rows) // 2
        assert all(r['route_distance'] > 0 for r in rows)
