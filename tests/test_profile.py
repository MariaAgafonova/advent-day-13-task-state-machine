import tempfile
import unittest
from pathlib import Path

from profile import JsonProfileRepository, UserProfile


class ProfileRepositoryTest(unittest.TestCase):
    def test_profile_is_saved_loaded_after_restart_and_partially_updated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles"
            repository = JsonProfileRepository(path)
            repository.create(
                UserProfile(
                    id="user_1",
                    name="Maria",
                    language="ru",
                    preferred_format="step_by_step",
                    interests=("Android",),
                ),
            )

            restarted = JsonProfileRepository(path)
            loaded = restarted.get("user_1")
            self.assertEqual(loaded.name, "Maria")
            self.assertEqual(loaded.preferred_format, "step_by_step")

            updated = restarted.update("user_1", {"responseLength": "detailed"})
            self.assertEqual(updated.response_length, "detailed")
            self.assertEqual(restarted.get("user_1").language, "ru")

    def test_profiles_are_isolated_by_user_id(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = JsonProfileRepository(directory)
            repository.create(UserProfile(id="alice", name="Alice", language="en"))
            repository.create(UserProfile(id="bob", name="Bob", language="ru"))

            self.assertEqual(repository.get("alice").name, "Alice")
            self.assertEqual(repository.get("bob").name, "Bob")
            self.assertNotEqual(repository.path_for_user("alice"), repository.path_for_user("bob"))

    def test_missing_profile_returns_defaults_without_writing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = JsonProfileRepository(directory)

            profile = repository.get_or_default("new-user")

            self.assertEqual(profile.id, "new-user")
            self.assertEqual(profile.language, "ru")
            self.assertFalse(Path(directory, "new-user.json").exists())

    def test_profile_payload_is_whitelisted_and_has_no_secret_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = JsonProfileRepository(directory)
            repository.create(UserProfile(id="safe", custom_instructions=("Use Kotlin",)))
            saved_text = Path(directory, "safe.json").read_text(encoding="utf-8")

            self.assertIn("customInstructions", saved_text)
            self.assertNotIn("api_key", saved_text)
            self.assertNotIn("token", saved_text)


if __name__ == "__main__":
    unittest.main()
