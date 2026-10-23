"""Keep upstream queue aggregation and action execution in one ordered transition."""

from importlib import import_module
import threading


def synchronized_client_class(base_client: type) -> type:
    """Wrap the pinned upstream client without importing robot dependencies here."""

    class SynchronizedRobotClient(base_client):
        def __init__(self, *args, **kwargs):
            self._queue_transition_lock = threading.RLock()
            super().__init__(*args, **kwargs)

        def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None):
            # Acquire this before the upstream queue/latest-action locks. Holding it
            # through send_action also prevents reinserting the action in flight.
            with self._queue_transition_lock:
                return super()._aggregate_action_queues(incoming_actions, aggregate_fn)

        def control_loop_action(self, verbose=False):
            with self._queue_transition_lock:
                return super().control_loop_action(verbose)

    return SynchronizedRobotClient


def main():
    upstream = import_module("lerobot.async_inference.robot_client")
    upstream.register_third_party_plugins()
    original_client = upstream.RobotClient
    upstream.RobotClient = synchronized_client_class(original_client)
    try:
        upstream.async_client()
    finally:
        upstream.RobotClient = original_client


if __name__ == "__main__":
    main()
