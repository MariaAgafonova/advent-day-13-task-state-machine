import unittest

from analytics import Analytics


class AnalyticsTest(unittest.TestCase):
    def test_snapshot_aggregates_main_and_auxiliary_usage(self):
        analytics = Analytics()
        analytics.record(
            {
                "strategy": "facts",
                "request_sent": True,
                "prompt_tokens": 80,
                "full_prompt_tokens": 120,
                "saved_tokens": 40,
                "savings_percent": 33.33,
                "completion_tokens": 20,
                "total_tokens": 100,
                "auxiliary_calls": 1,
                "auxiliary_total_tokens": 30,
                "total_tokens_including_auxiliary": 130,
                "main_cost_usd": 0.0001,
                "auxiliary_cost_usd": 0.00003,
                "cost_usd": 0.00013,
                "context_characters": 320,
                "full_context_characters": 480,
            },
            elapsed_seconds=0.25,
        )

        snapshot = analytics.snapshot()

        self.assertEqual(snapshot["total_requests"], 1)
        self.assertEqual(snapshot["total_prompt_tokens"], 80)
        self.assertEqual(snapshot["total_auxiliary_tokens"], 30)
        self.assertEqual(snapshot["total_tokens_including_auxiliary"], 130)
        self.assertEqual(snapshot["by_strategy"]["facts"]["cost_usd"], 0.00013)
        self.assertEqual(snapshot["series"][0]["elapsed_seconds"], 0.25)


if __name__ == "__main__":
    unittest.main()
