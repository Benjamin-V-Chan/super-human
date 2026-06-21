"""ScenarioAgent: ADL action -> trainable scene (deterministic library path)."""

from __future__ import annotations

import math

from prosthesis_rl.agents.scenario import ScenarioAgent, library_keys
from prosthesis_rl.contracts import ProblemSpec, ScenarioSpec
from prosthesis_rl.contracts.scenario import DEFAULT_REACH_M


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def test_tie_shoe_maps_to_floor_reach_scene():
    sc = ScenarioAgent().for_action("I want to tie my shoe")
    assert sc.task_id == "tie_shoe"
    assert sc.source == "library:tie_shoe"
    assert sc.posture == "floor_reach"
    # there is a shoe object and the hand has to go low (near the floor)
    assert any("shoe" in o.name for o in sc.objects)
    laces = sc.primary_waypoint()
    assert laces.pos[2] < 0.35  # the task point is low, not at chest height


def test_each_library_action_is_valid_and_reachable():
    agent = ScenarioAgent()
    probes = {
        "tie_shoe": "tie my shoe",
        "drink_bottle": "take a drink from the bottle",
        "open_drawer": "open the kitchen drawer",
        "open_jar": "unscrew the jar lid",
        "press_switch": "press the light switch",
        "open_door": "turn the door knob",
        "pick_from_floor": "pick the box up off the floor",
        "feeding": "eat from the bowl with a spoon",
    }
    for key, phrase in probes.items():
        sc = agent.for_action(phrase)
        assert sc.task_id == key, (phrase, sc.task_id)
        assert sc.validate() == []
        # every waypoint sits inside the arm's reach of the posture mount
        for wp in sc.waypoints:
            assert _dist(wp.pos, sc.mount_pos) <= DEFAULT_REACH_M + 1e-6


def test_unknown_action_falls_back_to_default_reach():
    # No library keyword and no valid LLM key -> the default reach scenario.
    sc = ScenarioAgent().for_action("contemplate the meaning of recursion")
    assert sc.task_id in {"reach_target", "llm_task"}
    assert sc.waypoints  # always something to reach
    assert sc.validate() == []


def test_primary_waypoint_is_highest_weight():
    sc = ScenarioAgent().for_action("open the drawer")
    primary = sc.primary_waypoint()
    assert primary.weight == max(w.weight for w in sc.waypoints)
    assert primary.name == "handle"


def test_scenario_json_roundtrips():
    sc = ScenarioAgent().for_action("drink from a water bottle")
    again = ScenarioSpec.from_json(sc.to_json())
    assert again.task_id == sc.task_id
    assert again.posture == sc.posture
    assert [o.name for o in again.objects] == [o.name for o in sc.objects]
    assert len(again.waypoints) == len(sc.waypoints)
    assert again.primary_waypoint().name == sc.primary_waypoint().name


def test_library_covers_the_core_adls():
    keys = set(library_keys())
    assert {"tie_shoe", "drink_bottle", "open_drawer"} <= keys


def test_derive_from_problem_spec_uses_primary_action():
    problem = ProblemSpec(
        primary_action="bend down to tie a shoelace",
        tasks=[{"id": "shoe_1", "name": "tie shoe"}],
    )
    sc = ScenarioAgent().derive(problem)
    assert sc.task_id == "shoe_1"          # keeps the problem's task id
    assert sc.source == "library:tie_shoe"
    assert sc.posture == "floor_reach"
