# SIEVE

**Situated Interpretation by Viable Elimination**

*Meaning is what survives.*

SIEVE is a deterministic mechanism for making a computer understand language
without neural networks, learned embeddings, statistical language models, or
training data. It does not translate a sentence into a parse tree and then into
logic. It does something more useful for an agent: it treats an utterance as a
set of constraints that eliminate incompatible interpretations of a grounded
situation.

The central claim is simple:

> Understanding is not producing one interpretation. It is maintaining the set
> of interpretations that remain possible, then reducing that set until action
> or answer is forced.

When more than one interpretation remains, SIEVE does not guess from
probabilities. It asks the smallest question that splits the surviving set. When
none remain, it reports the constraint that made the utterance impossible. When
exactly one remains, it acts or answers.

## Why this is different

Conventional language systems usually try to compress language into one of two
things: symbolic structures or vectors. Both approaches hide ambiguity inside a
model and then choose a likely reading.

SIEVE keeps ambiguity visible.

1. **A word is a sieve, not a label.** It is a rule that removes situations.
2. **A sentence is an intersection of sieves.** It is not first converted into a
   separate syntax tree.
3. **Ambiguity is resolved by discrimination, not probability.** If two readings
   survive, the system asks the question whose answer eliminates the most
   uncertainty.
4. **Unknown words are not guessed.** A word with no declared constraint is a
   genuine gap, not something to hallucinate.
5. **Execution is the test.** A command is understood when the surviving reading
   can be performed in the world model and produces the expected state.

## The mechanism

SIEVE works over a situation space: a typed graph of things, places, attributes,
relations, and executable actions.

For each word `w`, the lexicon defines a sieve:

```text
sigma(w) = the set of interpretations compatible with w
```

For an utterance `u = w1 w2 ... wn`:

```text
sigma(u) = sigma(w1) intersect sigma(w2) intersect ... intersect sigma(wn)
           intersect order-constraints(u)
```

The result is not a single parse. It is a set of live readings. SIEVE then runs:

```text
if |sigma(u)| == 0: report the violated constraint
if |sigma(u)| == 1: execute or answer the surviving reading
if |sigma(u)| > 1:  ask the question with the best worst-case split
```

The last rule is the important one. For each candidate question `q`, it computes
how many readings survive each possible answer. It asks the question that
minimizes the worst branch:

```text
q* = argmin_q max_v |{reading in sigma(u) : answer(reading, q) = v}|
```

In the prototype, the question is usually "Which cup do you mean: the red one or
the blue one?" The answer removes exactly half of the live readings, and the
next step either executes or asks the next discriminating question.

## What the prototype does

The repository contains a small, dependency-free demonstration world: two rooms,
objects with color and size, supports, containers, and a robot. It understands a
bounded *grounded* subset of English while recognizing broad English vocabulary:

- commands: `put`, `move`, `place`, `take`, `give`, `open`, `close`
- relations: `on`, `in`, `next to`, `to`, `from`; `in the living room` also
  reaches objects on its tables, shelves, floor, and inside containers
- questions: yes/no, existential (`is there ...`, `do you have ...`), `where`,
  `what is on/in ...`, `what color ...`, `what size ...`, `what are the
  contents of ...`, descriptions, inventories, `which ...`, and lists of a kind
- pronouns and ellipsis: `it`, `they`, `the red one`, `the one on the shelf`,
  and elliptical questions such as `how many are there`
- ambiguity handling: multiple objects, multiple destinations, sequential
  clarification; a clarification answer may select an option (`the red one`)
  or exclude one (`not the red one`)
- unknown-word repair: `what color is the car` offers to bind `car` to a kind
  (`car is a box`) or to a specific object (`call the red box a car`), then
  retries the original question; explicit `learn that ...` rules and freeform
  definitions can be saved across restarts; over-broad kinds such as
  `hat is a thing` are rejected instead of expanding into every object
- role-aware spelling recovery: common insertions, deletions, substitutions,
  and transpositions are corrected against the expected word category or the
  open English lexicon, with parse-based disambiguation and no guessing on
  dangerous near-matches
- broad lexical knowledge: 146,545 bundled WordNet 3.0 entries (63,783
  multiword), or 146,661 after the authored supplement is merged; the
  supplement adds function words and modern vocabulary, and `define`, `what
  does ... mean`, `how do you spell ...`, and definition-query typos are handled
- paraphrase normalization: contractions, polite requests, gerunds, embedded
  questions, `show me ...`, `list everything`, `tell me about ...`,
  `where can I find ...`, British `colour`, and common phrasal verbs
- conversation context: singular and plural pronouns, ellipsis, `there`, `here`,
  `back`, `the other one`, `what about ...`, `tell me more`, `what else ...`,
  bare references, repetition, and a typed turn transcript are resolved from
  explicit slots left by earlier turns; `what was my first question?` reads the
  transcript rather than guessing; ordinal, misspelled, and natural history
  paraphrases such as `what did I type first?` and `what's the first thing I
  said?` are normalized; `history` lists recent turns and `what did I say about
  the cup?` searches them without inventing a match
- conversational repair and memory: `what do you mean?`, `can you clarify`,
  `I'm confused`, `what was that again`, acknowledgements such as `I see,
  thanks`, `what do you remember`, `can you learn`, and basic identity
  questions are answered as dialogue instead of being treated as unknown nouns
- conversational meta-intent: bare known words (`clock`) get a definition and
  binding choices, `the red thing` asks a compact grounded question, `how do I
  bind it?` explains the rule, and criticism such as `why are you so stupid`
  is handled as feedback instead of being "corrected" into another word
- transparent explanation: `why` shows the live readings
- open concept universe: the bundled catalog exposes 146,661 lexical entries
  and 57,384 noun concepts; any recognized concept can be grounded on demand
  with `add a red elephant on the table`

Example:

```text
you> put the cup on the shelf
sieve> Which cup do you mean: the red one or the blue one?

you> why
sieve> I am holding 2 live interpretations:
  - put the red cup on the shelf
  - put the blue cup on the shelf
They differ on 'color'; the question chooses the smallest split.

you> the red one
sieve> Done. The red cup is now on the shelf.
```

Recognized English is not the same as grounded meaning:

```text
you> what color is the car
sieve> I recognize 'car' as English noun: a motor vehicle with four wheels;
       usually propelled by an internal combustion engine ... It is not bound
       in this world yet. Bind it to a kind, for example 'car is a box', or
       to a specific object, for example 'call the red box a car'.

you> call the red box a car
sieve> The red box is red.
```

The lexical layer knows that `serendipity`, `elephant`, and `smartphone` are
English words, and spelling recovery can repair `elephnat`. That does not give
the world a smartphone or an elephant. Recognition supplies lexical identity;
grounding supplies a referent or kind that actions and questions can use. The
same separation is why `what is the red cup` describes the grounded cup while
`what is an elephant` returns its dictionary definition.

Spelling errors are handled by the same elimination principle:

```text
you> pout the rde cuup onn the shelff
sieve> Done. The red cup is now on the shelf.

you> why
sieve> No ambiguity is pending. The current reading survived every constraint.
Spelling corrections: 'pout' -> 'put', 'rde' -> 'red', 'cuup' -> 'cup',
'onn' -> 'on', 'shelff' -> 'shelf'
```

The dictionary layer also has an authored table of common misspellings. It
repairs a phrase by testing nearby readings against the whole lexical phrase,
so `living rom` becomes `living room` and `prime numbr` becomes `prime number`
without silently changing unrelated words.

## Learning, history, and the concept universe

Learning is explicit and persistent, not gradient training. The CLI stores
rules in `~/.sieve/memory.json` (override with `--memory PATH` or
`SIEVE_MEMORY_PATH`). Every learned rule is inspectable with `knowledge`, and
`forget learning` erases the rules, created objects, world state, and history.

```text
you> learn that car means a box
sieve> Learned: 'car' behaves like the kind 'box'. Saved to ~/.sieve/memory.json.

you> what color is the car
sieve> Which box do you mean: the blue one or the red one?

you> the red one
sieve> The red box is red.
```

The same mechanism learns action, relation, and adjective aliases, plus
freeform definitions:

```text
you> learn that snag means take
sieve> Learned: 'snag' means the action 'take'.

you> learn that florp means a small container for tools
sieve> Learned: 'florp' means small container for tools.

you> add a florp in the blue box
sieve> Added the florp; it is in the blue box.
```

The chat transcript is typed and inspectable:

```text
you> history
sieve> Chat history (2 turns):
  1. you> where is the red cup
     sieve> The red cup is on the table.
  2. you> what color is the blue cup
     sieve> The blue cup is blue.

you> what did I say about the blue cup
sieve> The most recent turn about blue cup was: 'what color is the blue cup'.
```

History queries accept ordinary paraphrases (`what is 1 message`, `what did I
type first`, `what was my 2nd message`, `what's the first thing I said?`) and
repair contextual typos such as `whas`. They are marked as meta turns, so asking
about the first message does not move the record you are asking about. If you
paste terminal output, Windows prompts, `you>`/`sieve>` prefixes, and the startup
banner are stripped or ignored rather than stored as your words.

Questions about the mechanism are also explicit:

```text
you> what do you remember
sieve> I remember 1 explicit learned rule and 3 earlier chat turns. I do not
       store hidden weights or a hidden summary. Say 'knowledge' to inspect
       learned rules or 'history' to inspect the transcript.

you> can you learn
sieve> Yes, but not by gradient training. I learn explicit rules that you can
       inspect and erase. ...
```

The concept universe is as broad as the bundled lexicon, but only materialized
entities have physical locations. `universe` reports the catalog size:

```text
you> universe
sieve> Concept universe:
  lexical entries: 146,661
  definitions: 146,661 (100% coverage)
  noun concepts:  57,384
  verb concepts:  8,720
  adjective concepts: 20,749
  grounded objects: 12 (0 added by you)
```

This separation is intentional. Saying `what is an elephant` uses lexical
knowledge immediately; saying `add an elephant on the table` creates a grounded
entity that can then be moved and queried. The demo does not pretend that all
57,384 concepts are already physically present in the room.

Every bundled entry already has a definition, so it is not useful to issue
`learn that` for all 146,661 of them. `define X` and `what is X` already consult
the complete catalog. `learn that` is for a word outside the catalog, a
correction, or a world behaviour. Asking `learn every word` reports this instead
of copying the dictionary into `~/.sieve/memory.json`, which would add bulk but
no comprehension.

## Conversation context

Context is not a hidden summary produced by a model. It is a small set of
typed, inspectable slots: the last successful action or query, the salient
objects, the last destination, and the previous location of each object that
has moved. It also keeps a one-entry-per-user-message transcript, with meta
dialogue separated from substantive questions. A follow-up is interpreted by
filling missing slots from those values.

```text
you> put the red cup on the shelf
sieve> Done. The red cup is now on the shelf.

you> where is it
sieve> The red cup is on the shelf.

you> put it on the table
sieve> Done. The red cup is now on the table.

you> put it back
sieve> Done. The red cup is now on the shelf.

you> what about the blue cup
sieve> Done. The blue cup is now on the shelf.
```

`what about ...` reuses the previous frame. If the previous turn was a
question, it asks the same question about a new object or location. If it was
an action, it applies that action to the new object. `again` replays the
previous turn against the current world. `where did you put it?` asks about the
object of the last move, and `what did you do?` describes the last turn.
Plural follow-ups read the set of salient objects: after a count or list,
`where are they`, `what color are they`, and `how many are there` work without
restating the noun. `describe it`, `what else can you tell me`, and
`what else is on the table` extend the previous subject without a hidden
summary.

```text
you> what color is the red cup
sieve> The red cup is red.

you> wht abut the blu one
sieve> The blue cup is blue.

you> the other one
sieve> The red cup is red.

you> context
sieve> Conversation context:
  last question: the color of the red cup
  salient referents: the red cup
  last destination: on the shelf
  restorable locations: the red cup -> the table; the blue cup -> the table

you> what was my first question?
sieve> Your first question was: 'what color is the red cup?'

you> what did I ask before?
sieve> Your previous question was: 'what color is the red cup?'
```

Use `context` to inspect the slots and `forget` to clear them without resetting
the world. Context is committed only after a turn succeeds. A failed command or
an unresolved clarification question does not overwrite the previous turn.

## Run it

```powershell
python -m sieve.cli
python -m sieve.cli --demo
python -m sieve.cli --memory .\sieve-memory.json
python -m sieve.cli --no-memory
python -m sieve.cli --forget-learning
python -m unittest discover -s tests -v
```

The interactive CLI loads and saves learning memory by default. `--demo` uses an
isolated in-memory interpreter so the scripted transcript stays deterministic.
No third-party packages are required. Python 3.10 or newer is enough.

## What this proves and what it does not

This prototype proves the mechanism in a closed, grounded world. It shows that
ambiguity, pronouns, spatial reference, constraints, state changes, and unknown
words can be handled with no learned parameters at all.

It is not a claim that a hand-written lexicon can cover all human language. The
prototype uses frame assembly as a fast special case. A fuller SIEVE system would
replace that with a general constraint grammar and a planner, but the central
loop would stay the same: generate readings, eliminate them, and ask when
ambiguity remains.

The main limits are explicit:

- the ontology is small and hand-authored;
- open-domain words are recognized and defined, but only the bounded
  action/relation vocabulary can be executed; a known noun still needs a world
  binding or an `add ...` materialization before it can be moved or queried as
  an object;
- the concept universe is broad, but it is a lexical catalog, not an automatic
  simulation of 57,384 physical objects; materialized entities are explicit and
  persisted;
- misspellings are recovered when exactly one declared or lexical candidate is
  close enough; an ambiguous or genuinely new word enters the explicit
  definition loop rather than being guessed;
- phrase coverage is compositional plus explicit paraphrase rules. WordNet and
  the supplement do not enumerate every possible English sentence, idiom, or
  novel composition. There is no claim that this demo understands literally all
  English words and phrases: broad words are recognized, but grounded phrase
  coverage remains a finite, explicit set of rules;
- conversation context covers grounded anaphora, ellipsis, destination reuse,
  reversible moves, and same-turn repetition, but not open-ended narrative or
  arbitrary discourse structure;
- figurative language, long discourse, and world knowledge are not covered;
- the prototype asks questions rather than silently guessing, which is slower
  than a statistical model but far more inspectable.

The point is not to imitate a language model. The point is to show a different
primitive: **language understanding as the disciplined elimination of viable
interpretations, with questions as the only source of missing information.**

## Lexical data and attribution

`sieve/data/english_lexicon.json.gz` is a compact index built from WordNet 3.0.
WordNet is an authored lexical database, not a trained language model. Its
license is included in `LICENSE-WORDNET.txt`. The builder is
`tools/build_wordnet_lexicon.py`.

`sieve/data/common_english.json` is an authored supplement for function words,
conversational markers, and modern vocabulary that postdates WordNet 3.0. It is
small enough to inspect and edit directly; it contains no learned parameters.

See `MECHANISM.md` for the formal description and design rationale.
