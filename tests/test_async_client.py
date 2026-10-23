"""Exercise the pinned upstream queue methods without importing robot packages."""

import ast
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from so101_project.async_client import main, synchronized_client_class


@dataclass
class TimedAction:
    timestamp: float
    timestep: int
    action: float

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep

    def get_action(self):
        return self.action


def upstream_base():
    """Load the actual two methods, omitting dependency imports and constructors."""
    source = ROOT / "lerobot/src/lerobot/async_inference/robot_client.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    client = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RobotClient")
    names = {"_aggregate_action_queues", "control_loop_action"}
    methods = [node for node in client.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(methods) == 2
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *methods], type_ignores=[])
    namespace = {"Queue": Queue, "TimedAction": TimedAction, "time": time}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)

    class Base:
        def __init__(self):
            self.action_queue = Queue()
            self.action_queue_lock = threading.Lock()
            self.latest_action_lock = threading.Lock()
            self.latest_action = 0
            self.action_queue_size = []
            self.robot = SimpleNamespace(send_action=Mock(side_effect=lambda action: action))
            self.logger = Mock()

        def _action_tensor_to_action_dict(self, action):
            return {"joint.pos": action}

    for name in names:
        setattr(Base, name, namespace[name])
    return Base


class AsyncClientTests(unittest.TestCase):
    def actions(self, *timesteps):
        return [TimedAction(float(step), step, float(step)) for step in timesteps]

    def start_thread(self, callback):
        errors = []

        def run():
            try:
                callback()
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        return worker, errors

    def finish(self, *workers):
        for worker, errors in workers:
            worker.join(3)
            self.assertFalse(worker.is_alive(), "queue operation did not finish")
            self.assertEqual(errors, [])

    def test_preserves_weighted_average_future_steps_and_queue_sampling(self):
        client = synchronized_client_class(upstream_base())()
        client.latest_action = 4
        client.action_queue.put(TimedAction(5.0, 5, 50.0))
        client.action_queue.put(TimedAction(6.0, 6, 60.0))
        client._aggregate_action_queues(self.actions(3, 4, 5, 6, 7), lambda old, new: 0.3 * old + 0.7 * new)
        queued = list(client.action_queue.queue)
        self.assertEqual([item.timestep for item in queued], [5, 6, 7])
        self.assertEqual([item.action for item in queued], [18.5, 22.2, 7.0])
        self.assertEqual([item.timestamp for item in queued], [5.0, 6.0, 7.0])
        self.assertEqual(client.control_loop_action(), {"joint.pos": 18.5})
        self.assertEqual(client.latest_action, 5)
        self.assertEqual(client.action_queue_size, [3])

    def test_serializes_deque_iteration_and_consumption(self):
        client = synchronized_client_class(upstream_base())()
        entered, release = threading.Event(), threading.Event()
        attempted, consumed = threading.Event(), threading.Event()

        class PausingAction(TimedAction):
            def get_timestep(self):
                entered.set()
                if not release.wait(3):
                    raise TimeoutError("aggregation was not released")
                return self.timestep

        client.action_queue.put(PausingAction(1.0, 1, 1.0))
        client.action_queue.put(self.actions(2)[0])
        aggregate = self.start_thread(lambda: client._aggregate_action_queues(self.actions(1, 2)))
        self.assertTrue(entered.wait(3))

        def consume():
            attempted.set()
            client.control_loop_action()
            consumed.set()

        consumer = self.start_thread(consume)
        try:
            self.assertTrue(attempted.wait(3))
            self.assertFalse(consumed.wait(0.05))
        finally:
            release.set()
        self.finish(aggregate, consumer)
        self.assertEqual(client.latest_action, 1)
        self.assertEqual([item.timestep for item in client.action_queue.queue], [2])

    def test_does_not_reinsert_an_action_while_robot_is_executing_it(self):
        client = synchronized_client_class(upstream_base())()
        entered, release = threading.Event(), threading.Event()
        attempted, aggregated = threading.Event(), threading.Event()
        for action in self.actions(1, 2):
            client.action_queue.put(action)

        def send_action(action):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("robot action was not released")
            return action

        client.robot.send_action.side_effect = send_action
        consumer = self.start_thread(client.control_loop_action)
        self.assertTrue(entered.wait(3))

        def aggregate():
            attempted.set()
            client._aggregate_action_queues(self.actions(1, 2, 3))
            aggregated.set()

        receiver = self.start_thread(aggregate)
        try:
            self.assertTrue(attempted.wait(3))
            self.assertFalse(aggregated.wait(0.05))
        finally:
            release.set()
        self.finish(consumer, receiver)
        self.assertEqual(client.latest_action, 1)
        self.assertEqual([item.timestep for item in client.action_queue.queue], [2, 3])
        client.robot.send_action.assert_called_once_with({"joint.pos": 1.0})

    def test_entrypoint_reuses_upstream_lifecycle_and_restores_class_on_error(self):
        original = upstream_base()
        upstream = SimpleNamespace(RobotClient=original, register_third_party_plugins=Mock())

        def start():
            self.assertIsNot(upstream.RobotClient, original)
            self.assertTrue(issubclass(upstream.RobotClient, original))
            raise RuntimeError("synthetic shutdown")

        upstream.async_client = start
        with patch("so101_project.async_client.import_module", return_value=upstream) as import_module:
            with self.assertRaisesRegex(RuntimeError, "synthetic shutdown"):
                main()
        import_module.assert_called_once_with("lerobot.async_inference.robot_client")
        upstream.register_third_party_plugins.assert_called_once_with()
        self.assertIs(upstream.RobotClient, original)


if __name__ == "__main__":
    unittest.main()
