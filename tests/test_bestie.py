"""Offline tests: memory store + agent tool loop with a fake Nemotron client. Run: python -m unittest -v"""

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent  # noqa: E402
from memory import MemoryStore  # noqa: E402


class FakeClient:
    """Mimics openai.OpenAI().chat.completions.create with a scripted list of responses."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        msg = self.script.pop(0)
        return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=100, completion_tokens=20))


def tool_call(name, args, cid="c1"):
    return NS(content="", tool_calls=[NS(id=cid, function=NS(name=name, arguments=json.dumps(args)))])


def text(content):
    return NS(content=content, tool_calls=None)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "t.db")
        self.store = MemoryStore(self.db)

    def test_fact_crud_and_search(self):
        fid = self.store.add_fact("User's sister Priya loves orchids", tags="family")
        self.store.add_fact("User is vegetarian")
        self.assertEqual(self.store.add_fact("user's sister priya loves orchids"), fid)  # dedupe
        self.assertEqual(self.store.search_facts("what does my sister like?")[0]["id"], fid)
        self.store.update_fact(fid, "User's sister Priya loves tulips")
        self.assertIn("tulips", self.store.search_facts("priya")[0]["content"])
        self.assertTrue(self.store.delete_fact(fid))
        self.assertEqual(len(self.store.list_facts()), 1)

    def test_persists_across_instances(self):
        self.store.add_fact("Favorite color is teal")
        again = MemoryStore(self.db)
        self.assertEqual(again.search_facts("color")[0]["content"], "Favorite color is teal")

    def test_seed_skills_goals_wipe(self):
        self.assertTrue(self.store.get_skill("daily-plan"))
        gid = self.store.add_goal("Ship demo")
        self.store.set_goal_status(gid, "done")
        self.assertEqual(self.store.list_goals("open"), [])
        self.store.wipe_all(keep_skills=True)
        self.assertEqual(self.store.stats()["goals"], 0)
        self.assertGreater(self.store.stats()["skills"], 0)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(os.path.join(tempfile.mkdtemp(), "t.db"))

    def test_tool_loop_saves_memory(self):
        client = FakeClient([
            tool_call("save_memory", {"content": "User's dog is named Mochi"}),
            text("<think>ok</think>Got it — I'll remember Mochi!"),
        ])
        res = agent.run_turn(self.store, client, "nvidia/Nemotron-3_5-Lightning", "s1", "My dog is Mochi, remember that")
        self.assertEqual(res.reply, "Got it — I'll remember Mochi!")  # <think> stripped
        self.assertEqual(res.tool_events[0]["tool"], "save_memory")
        self.assertEqual(self.store.list_facts()[0]["content"], "User's dog is named Mochi")
        self.assertEqual(res.stats["calls"], 2)
        # Tool result was fed back to the model
        self.assertEqual(client.calls[1]["messages"][-1]["role"], "tool")

    def test_new_session_recalls_via_system_prompt(self):
        self.store.add_fact("User's dog is named Mochi")
        client = FakeClient([text("Your dog is Mochi.")])
        agent.run_turn(self.store, client, "m", "new-session", "What's my dog's name?")
        system = client.calls[0]["messages"][0]["content"]
        self.assertIn("Mochi", system)
        self.assertEqual(len(client.calls[0]["messages"]), 2)  # system + user, no old history

    def test_slash_commands_no_model(self):
        res = agent.run_turn(self.store, None, "m", "s", "/remember I take the train on Tuesdays")
        self.assertFalse(res.used_llm)
        res = agent.run_turn(self.store, None, "m", "s", "/recall train")
        self.assertIn("Tuesdays", res.reply)
        fid = self.store.list_facts()[0]["id"]
        agent.run_turn(self.store, None, "m", "s", f"/forget {fid}")
        self.assertEqual(self.store.list_facts(), [])

    def test_incognito_does_not_persist(self):
        client = FakeClient([text("hi")])
        agent.run_turn(self.store, client, "m", "s", "hello", persist=False)
        self.assertEqual(self.store.get_messages("s"), [])

    def test_bad_tool_args_dont_crash(self):
        client = FakeClient([
            NS(content="", tool_calls=[NS(id="x", function=NS(name="save_memory", arguments="{not json"))]),
            text("Sorry, retrying later."),
        ])
        res = agent.run_turn(self.store, client, "m", "s", "remember x")
        self.assertFalse(res.tool_events[0]["result"]["ok"])

    def test_skill_and_draft_tools(self):
        client = FakeClient([
            tool_call("run_skill", {"name": "daily-plan", "input": "busy day"}),
            text("1. Ship demo"),  # nested skill completion
            text("Here's your plan."),
        ])
        res = agent.run_turn(self.store, client, "m", "s", "run my daily plan")
        self.assertTrue(res.tool_events[0]["result"]["ok"])
        self.assertEqual(self.store.get_skill("daily-plan")["run_count"], 1)


if __name__ == "__main__":
    unittest.main()
