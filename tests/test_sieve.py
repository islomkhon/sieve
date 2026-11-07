import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sieve.engine import Interpreter
from sieve.memory import LearningMemory


class SieveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = Interpreter()

    def say(self, text: str):
        return self.engine.handle(text)

    def test_ambiguous_choice_then_action(self) -> None:
        result = self.say("put the cup on the shelf")
        self.assertEqual(result.kind, "clarify")
        self.assertIn("Which cup", result.text)
        self.assertIsNotNone(self.engine.pending)

        result = self.say("the red one")
        self.assertIn("red cup", result.text)
        self.assertEqual(self.engine.world.get("red_cup").location, "shelf")
        self.assertEqual(self.engine.world.get("blue_cup").location, "table")

    def test_negative_clarification_excludes_option(self) -> None:
        for answer, excluded_id, selected_id in (
            ("not the red one", "red_cup", "blue_cup"),
            ("no, not red", "red_cup", "blue_cup"),
            ("actually not the blue one", "blue_cup", "red_cup"),
        ):
            with self.subTest(answer=answer):
                self.engine = Interpreter()
                result = self.say("put the cup on the shelf")
                self.assertEqual(result.kind, "clarify")

                result = self.say(answer)
                self.assertEqual(result.kind, "answer")
                self.assertIn(selected_id.replace("_", " "), result.text)
                self.assertEqual(self.engine.world.get(selected_id).location, "shelf")
                self.assertEqual(self.engine.world.get(excluded_id).location, "table")

    def test_two_stage_ambiguity(self) -> None:
        result = self.say("put the cup in the box")
        self.assertIn("Which cup", result.text)

        result = self.say("the red one")
        self.assertIn("Which box", result.text)

        result = self.say("the blue one")
        self.assertIn("red cup", result.text)
        self.assertIn("blue box", result.text)
        self.assertEqual(self.engine.world.get("red_cup").location, "blue_box")

    def test_yes_no_question(self) -> None:
        result = self.say("is the red cup on the table")
        self.assertEqual(result.text, "Yes. The red cup is on the table.")

        result = self.say("is the red apple on the table")
        self.assertTrue(result.text.startswith("No."))
        self.assertIn("in the fridge", result.text)

    def test_list_question(self) -> None:
        result = self.say("what is on the table")
        self.assertIn("red cup", result.text)
        self.assertIn("blue cup", result.text)
        self.assertIn("green apple", result.text)
        self.assertTrue(result.text[0].isupper())

    def test_where_and_pronoun(self) -> None:
        result = self.say("where is the red apple")
        self.assertIn("in the fridge", result.text)

        self.say("where is the red cup")
        result = self.say("put it on the shelf")
        self.assertIn("red cup", result.text)
        self.assertEqual(self.engine.world.get("red_cup").location, "shelf")

    def test_color_question_is_resolved_by_discrimination(self) -> None:
        result = self.say("what color is the apple")
        self.assertIn("Which apple", result.text)

        result = self.say("the red one")
        self.assertIn("red apple", result.text)

    def test_unknown_noun_offers_repair(self) -> None:
        result = self.say("put the moon on the table")
        self.assertEqual(result.kind, "clarify")
        self.assertIn("moon", result.text)
        self.assertIn("recognize", result.text)
        self.assertIn("not bound", result.text)
        self.assertEqual(self.engine.world.get("red_cup").location, "table")

    def test_english_dictionary_definitions(self) -> None:
        result = self.say("define serendipity")
        self.assertEqual(result.kind, "answer")
        self.assertIn("serendipity", result.text)
        self.assertIn("fortunate", result.text)

        result = self.say("what does car mean")
        self.assertIn("motor vehicle", result.text)

    def test_open_vocabulary_typo_is_recovered(self) -> None:
        result = self.say("what color is the elephnat")
        self.assertEqual(result.kind, "clarify")
        self.assertIn("elephant", result.text)
        self.assertIn("not bound", result.text)

    def test_contractions_synonyms_and_phrasal_relations(self) -> None:
        result = self.say("Could you please put the blue cup on top of the shelf?")
        self.assertIn("blue cup", result.text)
        self.assertEqual(self.engine.world.get("blue_cup").location, "shelf")

        result = self.say("I'd like you to relocate the red apple into the refrigerator")
        self.assertIn("red apple", result.text)
        self.assertEqual(self.engine.world.get("red_apple").location, "fridge")

    def test_plural_question(self) -> None:
        result = self.say("where are the cups")
        self.assertEqual(result.kind, "clarify")
        self.assertIn("Which cup", result.text)

    def test_compound_request_uses_context(self) -> None:
        result = self.say("pick up the red cup and put it on the shelf")
        self.assertEqual(result.kind, "answer")
        self.assertIn("red cup", result.text)
        self.assertEqual(self.engine.world.get("red_cup").location, "shelf")

    def test_coordinated_request_uses_verb_ellipsis(self) -> None:
        result = self.say(
            "put the red cup on the shelf and the blue cup on the table"
        )
        self.assertEqual(result.kind, "answer")
        self.assertEqual(self.engine.world.get("red_cup").location, "shelf")
        self.assertEqual(self.engine.world.get("blue_cup").location, "table")

    def test_unknown_relation_is_not_guessed(self) -> None:
        result = self.say("put the red cup behind the table")
        self.assertEqual(result.kind, "error")
        self.assertIn("behind", result.text)

    def test_kind_definition_retries_original_question(self) -> None:
        result = self.say("what color is the car")
        self.assertEqual(result.kind, "clarify")
        self.assertIn("car", result.text)

        result = self.say("car is a box")
        self.assertIn("Which box", result.text)

        result = self.say("the red one")
        self.assertIn("red box", result.text)

    def test_object_definition_retries_original_question(self) -> None:
        result = self.say("what color is the car")
        self.assertEqual(result.kind, "clarify")

        result = self.say("call the red box a car")
        self.assertIn("red box", result.text)

        result = self.say("what color is the car")
        self.assertIn("red box", result.text)

    def test_definition_can_be_cancelled(self) -> None:
        result = self.say("what color is the car")
        self.assertEqual(result.kind, "clarify")

    def test_unknown_word_accepts_natural_elliptical_explanation(self) -> None:
        result = self.say("what color is the florp")
        self.assertEqual(result.kind, "clarify")
        self.assertIn("florp", result.text)

        result = self.say("it is a box")
        self.assertEqual(result.kind, "clarify")
        self.assertIn("Which box", result.text)

        result = self.say("the red one")
        self.assertIn("red box is red", result.text)

        result = self.say("cancel")
        self.assertEqual(result.kind, "info")
        self.assertIsNone(self.engine.definition_word)

        result = self.say("what color is the car")
        self.assertEqual(result.kind, "clarify")

    def test_impossible_destination_reports_constraint(self) -> None:
        result = self.say("put the red cup in the table")
        self.assertEqual(result.kind, "error")
        self.assertIn("not a container", result.text)
        self.assertEqual(self.engine.world.get("red_cup").location, "table")

    def test_container_state(self) -> None:
        result = self.say("is the fridge open")
        self.assertTrue(result.text.startswith("No."))

        self.say("open the fridge")
        result = self.say("is the fridge open")
        self.assertTrue(result.text.startswith("Yes."))

    def test_property_question(self) -> None:
        result = self.say("is the blue cup red")
        self.assertTrue(result.text.startswith("No."))
        self.assertIn("blue", result.text)

    def test_which_question(self) -> None:
        result = self.say("which cup is on the table")
        self.assertIn("red cup", result.text)
        self.assertIn("blue cup", result.text)

    def test_explain_pending_readings(self) -> None:
        self.say("put the cup on the shelf")
        result = self.say("why")
        self.assertIn("2 live interpretations", result.text)
        self.assertIn("red cup", result.text)
        self.assertIn("blue cup", result.text)

    def test_bare_adjective_answers_question(self) -> None:
        self.say("put the cup on the shelf")
        result = self.say("red")
        self.assertIn("red cup", result.text)
        self.assertEqual(self.engine.world.get("red_cup").location, "shelf")

    def test_spatial_reference_answers_question(self) -> None:
        result = self.say("put the box on the table")
        self.assertIn("Which box", result.text)

        result = self.say("the one on the shelf")
        self.assertIn("red box", result.text)
        self.assertEqual(self.engine.world.get("red_box").location, "table")

    def test_new_command_cancels_pending_choice(self) -> None:
        self.say("put the cup on the shelf")
        result = self.say("put the red apple on the table")
        self.assertIn("red apple", result.text)
        self.assertIsNone(self.engine.pending)

    def test_polite_prefix_is_ignored(self) -> None:
        result = self.say("can you put the red cup on the shelf")
        self.assertIn("red cup", result.text)
        self.assertEqual(self.engine.world.get("red_cup").location, "shelf")

    def test_misspelled_question_word_is_recovered(self) -> None:
        result = self.say("wht color is the red cup")
        self.assertEqual(result.kind, "answer")
        self.assertIn("red cup", result.text)

    def test_misspelled_command_is_recovered(self) -> None:
        result = self.say("pout the rde cuup onn the shelff")
        self.assertEqual(result.kind, "answer")
        self.assertIn("red cup", result.text)
        self.assertEqual(self.engine.world.get("red_cup").location, "shelf")

    def test_misspelled_state_question_is_recovered(self) -> None:
        result = self.say("is teh fridg opne")
        self.assertTrue(result.text.startswith("No."))
        self.assertIn("fridge", result.text)

    def test_misspelled_definition_retries_original(self) -> None:
        result = self.say("wht color is the car")
        self.assertEqual(result.kind, "clarify")

        result = self.say("car is a bx")
        self.assertIn("Which box", result.text)

        result = self.say("the rde one")
        self.assertIn("red box", result.text)

    def test_far_near_match_is_not_guessed(self) -> None:
        result = self.say("put the moon on the table")
        self.assertEqual(result.kind, "clarify")
        self.assertIn("moon", result.text)

        result = self.say("put the red cup behind the table")
        self.assertEqual(result.kind, "error")
        self.assertIn("behind", result.text)

    def test_why_reports_spelling_corrections(self) -> None:
        self.say("put the red cuup on the shelf")
        result = self.say("why")
        self.assertIn("cuup", result.text)
        self.assertIn("cup", result.text)

    def test_pronoun_and_back_follow_last_turn(self) -> None:
        self.say("put the red cup on the shelf")

        result = self.say("where is it")
        self.assertIn("red cup", result.text)
        self.assertIn("on the shelf", result.text)

        self.say("put it on the table")
        result = self.say("put it back")
        self.assertIn("red cup", result.text)
        self.assertEqual(self.engine.world.get("red_cup").location, "shelf")

    def test_there_reuses_last_destination(self) -> None:
        self.say("put the red cup on the shelf")
        result = self.say("put the blue cup there")
        self.assertIn("blue cup", result.text)
        self.assertEqual(self.engine.world.get("blue_cup").location, "shelf")

    def test_what_about_extends_previous_action(self) -> None:
        self.say("put the red cup on the shelf")
        result = self.say("what about the blue cup")
        self.assertIn("blue cup", result.text)
        self.assertEqual(self.engine.world.get("blue_cup").location, "shelf")

    def test_what_about_extends_previous_question(self) -> None:
        result = self.say("what color is the red cup")
        self.assertIn("red cup is red", result.text)

        result = self.say("what about the blue one")
        self.assertIn("blue cup is blue", result.text)

    def test_what_about_answers_pending_choice(self) -> None:
        result = self.say("put the cup on the shelf")
        self.assertIn("Which cup", result.text)

        result = self.say("what about the blue one")
        self.assertIn("blue cup", result.text)
        self.assertEqual(self.engine.world.get("blue_cup").location, "shelf")

    def test_other_one_is_relative_to_same_kind(self) -> None:
        self.say("where is the red cup")
        result = self.say("where is the other one")
        self.assertIn("blue cup", result.text)
        self.assertIn("on the table", result.text)

    def test_what_about_replaces_location_in_list_question(self) -> None:
        result = self.say("what is on the table")
        self.assertIn("red cup", result.text)

        result = self.say("what about the shelf")
        self.assertIn("red box", result.text)

    def test_what_about_chooses_relation_for_new_location(self) -> None:
        self.say("what is in the fridge")
        result = self.say("what about the shelf")
        self.assertIn("red box", result.text)
        self.assertIn("on the shelf", result.text)

    def test_again_repeats_last_question(self) -> None:
        first = self.say("where is the red cup")
        again = self.say("again")
        self.assertEqual(again.text, first.text)

    def test_again_repeats_last_action(self) -> None:
        self.say("put the red cup on the shelf")
        result = self.say("do it again")
        self.assertEqual(result.kind, "answer")
        self.assertIn("red cup", result.text)
        self.assertEqual(self.engine.world.get("red_cup").location, "shelf")

    def test_context_words_recover_spelling_errors(self) -> None:
        self.say("what color is the red cup")
        result = self.say("wht abut the blu one")
        self.assertIn("blue cup is blue", result.text)

    def test_context_only_advances_after_success(self) -> None:
        self.say("put the red cup on the shelf")
        failed = self.say("put the moon on the table")
        self.assertEqual(failed.kind, "clarify")

        self.assertIsNotNone(self.engine.context.last_action)
        action = self.engine.context.last_action
        self.assertIsNotNone(action)
        self.assertEqual(action.theme.id, "red_cup")
        result = self.say("context")
        self.assertIn("red cup", result.text)
        self.assertNotIn("moon", result.text)

    def test_forget_clears_context_without_resetting_world(self) -> None:
        self.say("put the red cup on the shelf")
        self.say("forget")
        self.assertIsNone(self.engine.context.last_action)
        self.assertEqual(self.engine.world.get("red_cup").location, "shelf")

        result = self.say("where is it")
        self.assertEqual(result.kind, "error")
        self.assertIn("it", result.text)

    def test_here_uses_robot_room(self) -> None:
        self.say("put the red cup here")
        self.assertEqual(self.engine.world.get("red_cup").location, "living_room")

    def test_back_history_is_kept_per_object(self) -> None:
        self.say("put the red cup on the shelf")
        self.say("put the red cup on the table")
        self.say("put the blue cup on the shelf")

        result = self.say("put the red cup back")
        self.assertIn("red cup", result.text)
        self.assertEqual(self.engine.world.get("red_cup").location, "shelf")

    def test_context_inspector_is_explicit(self) -> None:
        self.say("put the red cup on the shelf")
        result = self.say("context")
        self.assertIn("last action", result.text)
        self.assertIn("red cup", result.text)
        self.assertIn("last destination", result.text)
        self.assertIn("restorable locations", result.text)

    def test_bare_other_reference_extends_action(self) -> None:
        self.say("put the red cup on the shelf")
        result = self.say("the other one")
        self.assertIn("blue cup", result.text)
        self.assertEqual(self.engine.world.get("blue_cup").location, "shelf")

    def test_that_refers_to_last_question_subject(self) -> None:
        self.say("what color is the red cup")
        result = self.say("what about that")
        self.assertIn("red cup is red", result.text)

    def test_where_query_supplies_there_destination(self) -> None:
        self.say("where is the red cup")
        result = self.say("put the blue cup there")
        self.assertIn("blue cup", result.text)
        self.assertEqual(self.engine.world.get("blue_cup").location, "table")

    def test_where_did_you_put_it_uses_context(self) -> None:
        self.say("put the red cup on the shelf")
        result = self.say("where did you put it")
        self.assertIn("red cup", result.text)
        self.assertIn("on the shelf", result.text)

        result = self.say("where is it now")
        self.assertIn("on the shelf", result.text)

    def test_what_did_you_do_reports_last_action(self) -> None:
        self.say("put the red cup on the shelf")
        result = self.say("what did you do")
        self.assertIn("put the red cup on the shelf", result.text)

        result = self.say("what happened")
        self.assertIn("put the red cup on the shelf", result.text)

    def test_pending_question_can_be_repeated(self) -> None:
        first = self.say("put the cup on the shelf")
        self.assertEqual(first.kind, "clarify")

        repeated = self.say("which one")
        self.assertEqual(repeated.kind, "clarify")
        self.assertEqual(repeated.text, first.text)

        repeated = self.say("again")
        self.assertEqual(repeated.kind, "clarify")
        self.assertEqual(repeated.text, first.text)


    def test_description_frames_do_not_become_dictionary_lookups(self) -> None:
        for text in (
            "what is the red cup",
            "describe the red cup",
            "tell me about the red cup",
            "show me the red cup",
            "what does the red cup look like",
        ):
            with self.subTest(text=text):
                self.engine = Interpreter()
                result = self.say(text)
                self.assertEqual(result.kind, "answer")
                self.assertIn("The red cup is a cup", result.text)
                self.assertIn("on the table", result.text)
                self.assertNotIn("(noun)", result.text)

    def test_inventory_and_show_requests(self) -> None:
        result = self.say("list everything")
        self.assertIn("world contains", result.text)
        self.assertIn("red cup", result.text)
        self.assertIn("fridge", result.text)

        result = self.say("show me everything in the fridge")
        self.assertIn("red apple", result.text)
        self.assertIn("in the fridge", result.text)

        result = self.say("give me a list of things on the table")
        self.assertIn("red cup", result.text)
        self.assertIn("blue cup", result.text)

    def test_size_color_and_count_paraphrases(self) -> None:
        result = self.say("what colour is the red cup")
        self.assertIn("red cup is red", result.text)

        result = self.say("how big is the red cup")
        self.assertIn("red cup is small", result.text)

        result = self.say("what is the red cup's size")
        self.assertIn("red cup is small", result.text)

        result = self.say("how many cups are there on the table")
        self.assertIn("2 cups", result.text)

        result = self.say("how many red cups are on the table")
        self.assertIn("1 red cup", result.text)

    def test_embedded_questions_and_polite_gerunds(self) -> None:
        for text in (
            "can you tell me where the red cup is",
            "do you know where the red cup is",
            "where might the red cup be",
            "where can i find the red cup",
        ):
            with self.subTest(text=text):
                self.engine = Interpreter()
                result = self.say(text)
                self.assertIn("red cup is on the table", result.text)

        result = self.say("would you mind putting the red cup on the shelf")
        self.assertIn("red cup", result.text)
        self.assertEqual(self.engine.world.get("red_cup").location, "shelf")

    def test_separated_particle_and_count_filter(self) -> None:
        result = self.say("pick the red cup up")
        self.assertIn("holding the red cup", result.text)

        self.engine = Interpreter()
        result = self.say("how many red cups are on the table")
        self.assertIn("1 red cup", result.text)
        self.assertNotIn("2 cups", result.text)

    def test_dictionary_repairs_explicit_typos(self) -> None:
        result = self.say("definition of elephnat")
        self.assertIn("elephant", result.text)
        self.assertIn("read 'elephnat' as 'elephant'", result.text)

        result = self.say("define becausee")
        self.assertIn("because", result.text)
        self.assertIn("conjunction", result.text)

        result = self.say("define selfie")
        self.assertIn("photograph taken of oneself", result.text)

    def test_context_demonstrative_and_last_action_question(self) -> None:
        self.say("what color is the red cup")
        result = self.say("what is that")
        self.assertIn("red cup is a cup", result.text)

        self.say("put the red cup on the shelf")
        result = self.say("what did you put on the shelf")
        self.assertIn("put the red cup on the shelf", result.text)

    def test_conversational_frames(self) -> None:
        result = self.say("hello")
        self.assertIn("Hello", result.text)
        result = self.say("what is your name")
        self.assertIn("SIEVE", result.text)
        result = self.say("where am i")
        self.assertIn("living room", result.text)
        result = self.say("thank you")
        self.assertIn("welcome", result.text)

    def test_unknown_bare_word_enters_noun_repair(self) -> None:
        result = self.say("what is xyzzy")
        self.assertEqual(result.kind, "clarify")
        self.assertIn("xyzzy", result.text)
        self.assertIn("Bind it to a kind", result.text)

        self.engine = Interpreter()
        result = self.say("what is ubblax")
        self.assertIn("an 'ubblax'", result.text)

    def test_room_containment_and_counts(self) -> None:
        result = self.say("how many things are in the living room")
        self.assertIn("9 things", result.text)
        self.assertIn("in the living room", result.text)

        result = self.say("what objects are in the kitchen")
        self.assertIn("fridge", result.text)
        self.assertIn("red apple", result.text)
        self.assertIn("in the kitchen", result.text)

        result = self.say("put the red cup in the living room")
        self.assertIn("red cup", result.text)
        self.assertEqual(self.engine.world.get("red_cup").location, "living_room")

    def test_existential_and_contents_frames(self) -> None:
        cases = (
            ("is there a cup on the table", "2 cups"),
            ("are there any apples in the kitchen", "1 apple"),
            ("do you have a fridge", "1 fridge"),
            ("how many cups does the table have", "2 cups"),
            ("does the table have cups", "2 cups on the table"),
            ("what are the contents of the fridge", "red apple is in the fridge"),
            ("what're the contents of the fridge", "red apple is in the fridge"),
            ("give me the contents of the fridge", "red apple is in the fridge"),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.engine = Interpreter()
                result = self.say(text)
                self.assertEqual(result.kind, "answer")
                self.assertIn(expected, result.text)

    def test_context_followups_and_elaboration(self) -> None:
        self.say("what color is the red cup")
        result = self.say("what is its color")
        self.assertIn("red cup is red", result.text)

        result = self.say("how big is it")
        self.assertIn("red cup is small", result.text)

        result = self.say("tell me more")
        self.assertIn("More generally", result.text)
        self.assertIn("container", result.text)

        result = self.say("what else is on the table")
        self.assertIn("blue cup", result.text)
        self.assertIn("green apple", result.text)
        self.assertNotIn("red cup", result.text)

    def test_room_context_and_deictic_locations(self) -> None:
        result = self.say("what is in the kitchen")
        self.assertIn("red apple", result.text)

        result = self.say("what about in the living room")
        self.assertIn("red cup", result.text)
        self.assertIn("in the living room", result.text)

        result = self.say("and on the shelf")
        self.assertIn("red box", result.text)
        self.assertIn("on the shelf", result.text)

        result = self.say("what is in here")
        self.assertIn("living room", result.text)

        result = self.say("what is in this room")
        self.assertIn("living room", result.text)

    def test_dictionary_phrase_repair_and_spelling(self) -> None:
        result = self.say("define recieve")
        self.assertIn("receive (verb)", result.text)
        self.assertIn("read 'recieve' as 'receive'", result.text)

        result = self.say("define living rom")
        self.assertIn("living room (noun)", result.text)

        result = self.say("what is a prime numbr")
        self.assertIn("prime number (noun)", result.text)

        result = self.say("how do you spell elephnat")
        self.assertIn("e-l-e-p-h-a-n-t", result.text)

        result = self.say("what does teh word serendipity mean")
        self.assertIn("serendipity (noun)", result.text)
        self.assertIn("fortunate discoveries", result.text)

        result = self.say("define now")
        self.assertIn("now (noun, adverb)", result.text)

    def test_separated_particles_and_room_question(self) -> None:
        result = self.say("switch the fridge on")
        self.assertIn("now open", result.text)
        self.assertTrue(self.engine.world.get("fridge").attrs["open"])

        result = self.say("turn the fridge off")
        self.assertIn("now closed", result.text)
        self.assertFalse(self.engine.world.get("fridge").attrs["open"])

        result = self.say("turn the fridge back on")
        self.assertIn("now open", result.text)

        result = self.say("shut the fridge down")
        self.assertIn("now closed", result.text)

        result = self.say("what room is the red cup in")
        self.assertIn("on the table", result.text)

        result = self.say("which room is the fridge in")
        self.assertIn("in the kitchen", result.text)

        result = self.say("what is the color of the red cup")
        self.assertIn("red cup is red", result.text)

        result = self.say("what cups are on the table")
        self.assertIn("red cup", result.text)
        self.assertIn("blue cup", result.text)

    def test_list_kind_requests(self) -> None:
        result = self.say("show me all the cups")
        self.assertIn("2 cups", result.text)
        self.assertIn("red cup", result.text)
        self.assertIn("blue cup", result.text)

        result = self.say("list cups")
        self.assertIn("red cup", result.text)
        self.assertIn("blue cup", result.text)

        result = self.say("can you show me all the cups")
        self.assertIn("2 cups", result.text)

    def test_plural_context_followups(self) -> None:
        result = self.say("how many cups are on the table")
        self.assertIn("2 cups", result.text)

        result = self.say("where are they")
        self.assertIn("red cup is on the table", result.text)
        self.assertIn("blue cup is on the table", result.text)

        result = self.say("what color are they")
        self.assertIn("red cup is red", result.text)
        self.assertIn("blue cup is blue", result.text)

        result = self.say("how many are there")
        self.assertIn("2 cups", result.text)

        result = self.say("how many are on the table")
        self.assertIn("2 cups on the table", result.text)

    def test_description_and_removal_followups(self) -> None:
        self.say("where is the red cup")

        result = self.say("describe it")
        self.assertIn("red cup is a cup", result.text)
        self.assertIn("on the table", result.text)

        result = self.say("is it still red")
        self.assertIn("red cup is red", result.text)

        result = self.say("what else can you tell me")
        self.assertIn("More generally", result.text)

        result = self.say("take it down")
        self.assertIn("holding the red cup", result.text)
        self.assertEqual(self.engine.world.get("red_cup").location, "robot")

    def test_reported_transcript_gets_specific_answers(self) -> None:
        result = self.say("put red hat on the shelf")
        self.assertEqual(result.kind, "clarify")
        self.assertIn("hat", result.text)
        self.assertIn("not bound", result.text)

        result = self.say("why should I call the hat as a box?")
        self.assertEqual(result.kind, "info")
        self.assertIn("do not have to", result.text)
        self.assertNotIn("who", result.text.lower())
        self.assertFalse(
            any(
                correction.original == "why" for correction in self.engine.parser.corrections
            )
        )

        result = self.say("hat is a thing")
        self.assertEqual(result.kind, "error")
        self.assertIn("too broad", result.text)
        self.assertNotIn("Which one", result.text)
        self.assertIsNone(self.engine.pending)
        self.assertNotIn("hat", self.engine.noun_aliases)
        self.assertEqual(self.engine.world.get("red_cup").location, "table")

        result = self.say("why")
        self.assertIn("too broad", result.text)

        result = self.say("what is the hat and how to bind it?")
        self.assertEqual(result.kind, "info")
        self.assertIn("binding", result.text.lower())
        self.assertIn("hat", result.text)
        self.assertNotIn("who", result.text.lower())

    def test_bare_known_noun_and_bare_description(self) -> None:
        result = self.say("clock")
        self.assertEqual(result.kind, "clarify")
        self.assertIn("clock", result.text)
        self.assertIn("timepiece", result.text)
        self.assertEqual(self.engine.definition_word, "clock")

        result = self.say("clock is a box")
        self.assertEqual(result.kind, "info")
        self.assertIn("clock", result.text)
        self.assertEqual(self.engine.noun_aliases["clock"], "box")

        result = self.say("the red thing")
        self.assertEqual(result.kind, "clarify")
        self.assertIn("red cup", result.text)
        self.assertIn("red apple", result.text)
        self.assertIn("red box", result.text)
        self.assertNotIn("I understand commands", result.text)

        result = self.say("the red cup")
        self.assertEqual(result.kind, "answer")
        self.assertIn("red cup is a cup", result.text)

    def test_history_answers_first_and_previous_questions(self) -> None:
        self.say("where is the red cup")
        self.say("what color is the blue cup")

        result = self.say("what was my first question?")
        self.assertEqual(result.kind, "info")
        self.assertIn("where is the red cup", result.text)

        result = self.say("what did I ask before?")
        self.assertIn("what color is the blue cup", result.text)

    def test_explicit_learning_persists_across_interpreter_sessions(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            first = Interpreter(memory=LearningMemory(path=path))
            result = first.handle("learn that car means a box")
            self.assertEqual(result.kind, "info")
            self.assertTrue(path.exists())

            second = Interpreter(memory=LearningMemory(path=path))
            result = second.handle("what color is the car")
            self.assertEqual(result.kind, "clarify")
            self.assertIn("Which box", result.text)

            result = second.handle("the red one")
            self.assertIn("red box is red", result.text)

    def test_learned_action_alias_is_used_by_later_commands(self) -> None:
        result = self.say("learn that snag means take")
        self.assertEqual(result.kind, "info")

        result = self.say("snag the red cup")
        self.assertIn("holding the red cup", result.text)
        self.assertEqual(self.engine.world.get("red_cup").location, "robot")

    def test_direct_definition_without_learn_prefix(self) -> None:
        result = self.say("car is a box")
        self.assertEqual(result.kind, "info")
        self.assertIn("Learned", result.text)

        result = self.say("what color is the car")
        self.assertEqual(result.kind, "clarify")
        self.assertIn("Which box", result.text)

    def test_learned_freeform_definition_is_answered_and_materialized(self) -> None:
        result = self.say("learn that florp means a small container for tools")
        self.assertEqual(result.kind, "info")

        result = self.say("what is a florp")
        self.assertIn("learned", result.text)
        self.assertIn("container for tools", result.text)

        result = self.say("add a florp in the blue box")
        self.assertEqual(result.kind, "answer")
        self.assertIn("florp", result.text)

        result = self.say("put the red cup in the florp")
        self.assertEqual(self.engine.world.get("red_cup").location, "florp")

    def test_open_vocabulary_objects_persist(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            first = Interpreter(memory=LearningMemory(path=path))
            result = first.handle("add a red elephant on the table")
            self.assertEqual(result.kind, "answer")

            second = Interpreter(memory=LearningMemory(path=path))
            result = second.handle("where is the red elephant")
            self.assertIn("red elephant is on the table", result.text)

    def test_history_command_and_topic_lookup(self) -> None:
        self.say("where is the red cup")
        self.say("what color is the blue cup")

        result = self.say("history")
        self.assertIn("where is the red cup", result.text)
        self.assertIn("what color is the blue cup", result.text)

        result = self.say("what did I say about the blue cup")
        self.assertIn("what color is the blue cup", result.text)

        result = self.say("what did you say about the blue cup")
        self.assertIn("blue cup is blue", result.text)

    def test_history_accepts_plain_ordinal_and_misspelled_phrasings(self) -> None:
        self.say("hi")
        self.say("what is on the table")

        result = self.say("what is 1 message")
        self.assertIn("first message", result.text)
        self.assertIn("'hi'", result.text)

        result = self.say("what is the first message")
        self.assertIn("first message", result.text)
        self.assertIn("'hi'", result.text)

        result = self.say("what whas my first message")
        self.assertIn("first message", result.text)
        self.assertIn("'hi'", result.text)

        result = self.say("what was the second message")
        self.assertIn("second message", result.text)
        self.assertIn("what is on the table", result.text)

        result = self.say("what was my 2nd message")
        self.assertIn("second message", result.text)
        self.assertIn("what is on the table", result.text)

        result = self.say("what's the first thing I said")
        self.assertIn("first message", result.text)
        self.assertIn("'hi'", result.text)

    def test_acknowledgements_and_demo_command_are_conversational(self) -> None:
        result = self.say("I see")
        self.assertEqual(result.kind, "info")
        self.assertNotIn("not bound", result.text)

        result = self.say("I see, thanks")
        self.assertEqual(result.kind, "info")
        self.assertNotIn("not bound", result.text)

        result = self.say("got it, thank you")
        self.assertEqual(result.kind, "info")
        self.assertNotIn("not bound", result.text)

        result = self.say("perfect")
        self.assertEqual(result.kind, "info")
        self.assertNotIn("tense of verbs", result.text)

        result = self.say("okay")
        self.assertEqual(result.kind, "info")

        result = self.say("demo")
        self.assertEqual(result.kind, "info")
        self.assertIn("--demo", result.text)

    def test_lexical_comparison_questions(self) -> None:
        for text in (
            "what is the difference between phone and telephone",
            "compare phone and telephone",
            "how is a phone different from a telephone",
            "phone vs telephone",
            "are phone and telephone the same",
        ):
            with self.subTest(text=text):
                result = self.say(text)
                self.assertEqual(result.kind, "answer")
                self.assertIn("phone", result.text)
                self.assertIn("telephone", result.text)
                self.assertNotIn("not bound", result.text)

    def test_clarification_variants_reference_the_last_answer(self) -> None:
        result = self.say("where is the red cup")
        self.assertIn("on the table", result.text)

        for text in ("can you clarify", "I'm confused", "what do you mean by that"):
            with self.subTest(text=text):
                result = self.say(text)
                self.assertEqual(result.kind, "info")
                self.assertIn("red cup", result.text)
                self.assertNotIn("am", result.text.lower().split())

        result = self.say("huh")
        self.assertIn("My last answer was", result.text)
        self.assertIn("red cup", result.text)

        self.engine = Interpreter()
        result = self.say("what do you mean")
        self.assertEqual(result.kind, "info")
        self.assertIn("not answered anything yet", result.text)

    def test_history_does_not_invent_a_topic_match(self) -> None:
        self.say("hi")

        result = self.say("what did I say about calculus")
        self.assertEqual(result.kind, "info")
        self.assertIn("found no earlier turn", result.text)
        self.assertNotIn("'hi'", result.text)

    def test_pasted_cli_transcript_lines_do_not_pollute_history(self) -> None:
        ignored = (
            r"D:\newFormula>python -m sieve.cli",
            "Sieve demo. Type 'help' for examples, 'quit' to exit.",
            "No model, no training, no embeddings.",
            r"Learning memory: C:\Users\islom\.sieve\memory.json",
            "sieve> Hello.",
        )
        for text in ignored:
            with self.subTest(text=text):
                result = self.say(text)
                self.assertEqual(result.kind, "info")
                self.assertIn("Ignored", result.text)
        self.assertEqual(self.engine.context.turns, [])

        result = self.say("you> hi")
        self.assertIn("Hello", result.text)
        self.assertEqual(len(self.engine.context.turns), 1)
        self.assertEqual(self.engine.context.turns[0].user, "hi")

    def test_memory_and_identity_meta_questions(self) -> None:
        for text in ("learn every word", "learn the whole dictionary"):
            with self.subTest(text=text):
                result = self.say(text)
                self.assertEqual(result.kind, "info")
                self.assertIn("already have definitions", result.text)
                self.assertIn("100% coverage", result.text)
                self.assertEqual(self.engine.learned_definitions, {})

        result = self.say("can you learn")
        self.assertIn("explicit rules", result.text)
        self.assertNotIn("Use a form", result.text)

        result = self.say("what do you remember")
        self.assertIn("no explicit rules", result.text)

        self.say("learn that snag means take")
        self.say("hi")
        result = self.say("what do you remember")
        self.assertIn("1 explicit learned rule", result.text)
        self.assertIn("1 earlier chat turn", result.text)

        result = self.say("do you remember what I said")
        self.assertIn("You last said: 'hi'", result.text)

        result = self.say("who made you")
        self.assertIn("model-free interpreter", result.text)

        result = self.say("are you alive")
        self.assertIn("deterministic software", result.text)

    def test_universe_reports_lexical_and_grounded_counts(self) -> None:
        result = self.say("universe")
        self.assertEqual(result.kind, "info")
        self.assertIn("noun concepts", result.text)
        self.assertIn("146,", result.text)
        self.assertIn("100% coverage", result.text)

    def test_reset_keeps_learned_rules_but_clears_added_objects(self) -> None:
        self.say("learn that car means a box")
        self.say("add an elephant on the table")
        self.say("reset")

        result = self.say("what color is the car")
        self.assertEqual(result.kind, "clarify")
        self.assertIn("Which box", result.text)

        result = self.say("where is the elephant")
        self.assertEqual(result.kind, "clarify")
        self.assertIn("not bound", result.text)

    def test_criticism_gets_a_useful_reply_not_a_who_question(self) -> None:
        result = self.say("why are you so stupid")
        self.assertEqual(result.kind, "info")
        self.assertIn("fair criticism", result.text)
        self.assertNotIn("who", result.text.lower())

        self.engine = Interpreter()
        result = self.say("what color is the bad apple")
        self.assertEqual(result.kind, "error")
        self.assertNotIn("fair criticism", result.text)

    def test_valid_past_auxiliary_and_more_paraphrases(self) -> None:
        result = self.say("what was on the table")
        self.assertEqual(result.kind, "answer")
        self.assertIn("red cup", result.text)
        self.assertNotIn("bivalent metallic element", result.text)

        cases = (
            ("make the fridge open", "now open"),
            ("how many cups are sitting on the table", "2 cups"),
            ("what have you got in the fridge", "red apple"),
            ("what were the contents of the fridge", "red apple"),
            ("the red apple is where", "in the fridge"),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.engine = Interpreter()
                result = self.say(text)
                self.assertEqual(result.kind, "answer")
                self.assertIn(expected, result.text)


if __name__ == "__main__":
    unittest.main()
