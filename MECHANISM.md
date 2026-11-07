# The SIEVE Mechanism

## 1. Premise

Most language systems try to answer one question directly:

> Which meaning did the speaker intend?

SIEVE answers a different question:

> Which meanings are still compatible with everything observed so far?

This changes the object the computer manipulates. A meaning is not a single
symbol, vector, or parse tree. It is a **set of surviving interpretations**. The
computer understands an utterance when that set is small enough to act on, and
it resolves the rest by asking questions.

The name is a backronym: **Situated Interpretation by Viable Elimination**.

## 2. Situation space

A situation space is a typed graph:

```text
S = (E, A, R, L, W)

E = entities
A = attributes
R = relations
L = executable actions
W = current world state
```

An **interpretation** is a binding of linguistic material to this graph:

```text
i = (binding, reading, cost)
```

For a command, a reading is an action with grounded arguments. For a question,
it is a query with grounded arguments. A binding maps noun phrases, pronouns,
and spatial descriptions to entities in `E`.

The key property is that interpretations are enumerable and inspectable. Nothing
is hidden in a weight matrix.

## 3. Sieve semantics

Each lexical item `w` contributes a sieve:

```text
sigma(w) subset of I
```

For example, in the prototype:

```text
sigma("cup")      = interpretations whose theme has kind cup
sigma("red")      = interpretations whose focused entity has color red
sigma("on")       = interpretations where the theme is supported by the object
sigma("shelf")    = interpretations whose location is the shelf
sigma("put")      = interpretations of a placement action
sigma("it")       = interpretations whose referent is the current salient entity
```

A word is not a dictionary gloss. It is a predicate over possible readings.

For an utterance `u = w1 ... wn`, the sieve is the intersection:

```text
sigma(u) = intersection over k of sigma(wk)
           intersect order-constraints(u)
           intersect world-constraints(u)
```

`order-constraints` encode adjacency, role order, and attachment. In a full
system these would be general constraints over token roles. In the prototype
they are implemented by a small frame assembler. Either way, syntax is not a
separate translation stage; it is one more source of constraints.

The set `sigma(u)` is the **partition** of the situation space induced by the
utterance. This is the formal replacement for "the meaning of the sentence."
Two utterances are treated as equivalent in a context when they leave the same
set of interpretations alive.

## 4. Binding without a parse tree

SIEVE does not need to choose a tree before it can reason. It needs to choose a
binding. The prototype implements binding as deterministic frame assembly:

1. recognize a verb frame;
2. resolve noun phrases to candidate entities;
3. resolve prepositional phrases to candidate destinations;
4. take the Cartesian product of candidates;
5. reject readings that violate selectional or world constraints.

The Cartesian product is the "possibility cloud." It is deliberately small in a
bounded world. A general implementation would use constraint propagation and
arc consistency to avoid materializing the full product.

This is why the mechanism is different from "parse first, disambiguate later."
Ambiguity is present at the level of entities and actions from the beginning,
which is exactly the level at which it matters to an agent.

### Spelling as a lexical sieve

A misspelling is treated as ambiguity over lexical identity, not as a separate
preprocessing trick. When a token is not declared, SIEVE generates the nearest
known words **in the grammatical category expected at that position**:

- a question position may map `wht` to `what` or `who`;
- a noun position may map `cuup` to `cup`;
- a relation position may map `onn` to `on`.

The edit budget is role-independent but length-aware: one edit for short words,
two for medium words, three for long words. Candidates outside the budget are
not considered. If exactly one candidate survives, it is used. If several
candidates survive, each candidate is substituted into the sentence and parsed;
the candidate that yields a valid structure wins. If that still leaves more
than one reading, the system refuses to guess and falls through to explicit
repair.

This prevents dangerous near-matches. For example, `moon` is not silently
changed to `room`, and `beside` is not silently changed to `inside`. Those are
genuinely unknown or unsupported words, so the user is asked to define them.
Spelling recovery maps a typo onto existing vocabulary; it does not invent new
meaning.

The category check has two layers. Declared world words are tried first, so a
typo near an executable word resolves without scanning the open vocabulary.
If no declared candidate survives, SIEVE searches WordNet lemmas in the part of
speech expected at that position. This is why `elephnat` can be recognized as
`elephant` and reported as an ungrounded English noun, while `moon` is not
silently rewritten into the grounded word `room`. If several lexical candidates
remain at the best edit distance, SIEVE refuses to guess.

An authored `COMMON_MISSPELLINGS` table covers frequent human typos such as
`recieve`, `teh`, and `becuase`. Dictionary lookup also repairs a whole phrase
by testing nearby component spellings together, which is why `living rom`
resolves to `living room` and `prime numbr` to `prime number` without changing
unrelated words.

### Lexical breadth without learned parameters

The vocabulary layer is data, not a model:

```text
sieve/data/english_lexicon.json.gz   146,545 bundled WordNet 3.0 entries
                                    (63,783 multiword terms)
sieve/data/common_english.json      authored function-word and modern supplement
```

It supplies spelling identity, parts of speech, irregular forms, multiword
terms, and short definitions. All 146,661 bundled entries have a definition, so
`define`, `meaning of`, `what does X mean`, and `what is the definition of X`
consult that data directly; there is no need to copy the catalog into learned
rules. Explicit definition queries may repair an unknown token only when one
lexical candidate is unambiguous.

Recognizing a word is deliberately separate from grounding it. A known noun
can be defined immediately, but it still needs a kind or object binding before
the action and question frames can use it. `what is an elephant` is a lexical
question; `what is the red cup` is a grounded description because the cup
already exists in the world.

Surface breadth is a separate explicit layer. Contractions, polite prefixes,
gerunds, separated particles, embedded questions, British spellings, common
phrasal verbs, and requests such as `show me`, `list everything`, or `where can I
find` are rewritten into canonical frames. Existential frames (`is there a cup`,
`do you have a fridge`), content questions (`what are the contents of ...`),
room containment, and kind listings (`show me all the cups`) are explicit
frames rather than special replies. These rules cover many phrasings, but they
are rules, not a claim to enumerate every possible English sentence.

## 5. Discrimination

Suppose `n > 1` readings survive. The system has three choices:

1. guess using a statistical prior;
2. act on all readings (usually wrong);
3. ask a question.

SIEVE always asks.

Each candidate question `q` is a function from interpretations to answers:

```text
q: I -> V
```

For each possible answer `v`, let `M_v` be the surviving set after `v`. The
question's worst branch is:

```text
worst(q) = max over v of |M_v|
```

The chosen question minimizes the worst branch:

```text
q* = argmin_q worst(q)
```

Ties are broken by a question cost: length, awkwardness, and the number of
options. This is a minimax information-gathering policy, not a probability
model.

In the prototype:

- two cups with different colors -> ask about color;
- two objects of different kinds -> ask which kind;
- two locations -> ask which location;
- two destination objects -> ask which destination after the theme is fixed.

The dialogue state is `(S, M, pending)`, where `pending` records the surviving
readings, the slot being discriminated, and the question asked. A follow-up such
as "the red one" is itself a sieve over the pending set. A negative follow-up
such as "not the red one" removes the named readings from that set instead of
selecting them.

## 6. Conversation context as reversible slots

Context is not a compressed summary produced by a model. It is a tuple of
explicit slots:

```text
C = (E_salient, A_last, Q_last, D_last, L_saved, T_turns, R_reject)

E_salient = objects made salient by the last successful turn
A_last    = the last successfully executed action
Q_last    = the last successfully answered query
D_last    = the most recently grounded destination
L_saved   = previous location of each object that has moved
T_turns   = one recorded user/assistant turn per top-level message
R_reject  = the reason the last request was rejected, if any
```

Every context rule reads or writes those slots. No rule uses similarity,
frequency, or a learned prior.

- `it` / `them` / `they` bind to `E_salient`; a plural question is answered
  over every salient object rather than collapsed to one choice.
- `one` first searches the kind of `E_salient`; if that kind has no match, it
  widens to all movable objects, then reapplies the adjectives and modifiers.
- `the other one` selects an object of the same kind as `E_salient`, excluding
  the salient objects. If that leaves more than one, SIEVE asks instead of
  guessing.
- `there` copies `D_last`.
- `back` looks up the theme in `L_saved` and creates a destination from that
  saved location.
- `here` resolves to the robot's current location.
- `what about X`, `and X`, and a bare reference clone the previous frame: the
  same query with a new argument, or the same action with a new theme. If `X`
  names a location, a location query substitutes the destination instead.
- `where did you put it?` rewrites to a location query about the last move,
  while `what did you do?` and `what happened?` describe the last action.
- `describe it`, `tell me more`, `what else can you tell me`, and
  `what else is on the table` reuse the salient subject. `what else` also
  excludes subjects already listed by the previous answer.
- Elliptical count questions such as `how many are there` and `how many are on
  the table` copy the noun and modifiers from `Q_last`.
- `again` replays `A_last` or `Q_last` against the current world.
- `what was my first question?`, `what did I ask before?`, and `what did I say
  first?` read `T_turns`. History queries are themselves marked as meta turns,
  so they do not displace the substantive question being asked about. Plain
  ordinals (`1`, `1st`-`5th`), `thing` phrasing, and contextual typos such as
  `whas` are normalized before lookup.
- `history` renders recent entries from `T_turns`; `what did I say about X?`
  filters that transcript by an explicit keyword, and `what did you say?`
  reads the assistant side of the most recent turn. If no turn contains the
  requested keyword, the reply says so instead of falling back to an unrelated
  recent turn.
- `can you clarify`, `what do you mean?`, `I'm confused`, and `what was that
  again?` consult the last substantive answer; acknowledgements such as `I see,
  thanks` are dialogue moves, not noun-definition failures.
- `what do you remember`, `can you learn`, `who made you`, and `are you alive?`
  report the actual mechanism: explicit rules, typed turns, and deterministic
  software rather than hidden weights or consciousness.
- Pasted terminal transport noise (`C:\path>`, `you>`, `sieve>`, and the startup
  banner) is stripped or ignored before a turn is recorded in `T_turns`.
- Bare known nouns (`clock`) consult the lexical database and enter the binding
  dialogue. Bare descriptions (`the red thing`) are grounded against the world;
  if several objects match, SIEVE asks a compact choice question.
- Meta-intent rules answer `how do I bind it?`, explain what a kind binding
  means, and treat criticism as feedback. They run before compound splitting so
  an `and` inside a question is not mistaken for a second command.

The context is committed only after a reading survives grounding, validation,
and execution or answering. A failed command, an unresolved pending question,
or a definition request leaves the slots unchanged. This prevents a repair
dialogue from corrupting the referents of the previous successful turn.

`context` prints the slots and `forget` clears them without changing the world.

### Explicit learning and persistent memory

The same principle extends across sessions. There is no gradient update. A user
supplies a rule, and the rule is stored as JSON:

```text
learn that car means a box        -> noun kind alias
learn that snag means take        -> action alias
learn that florp means a container-> freeform definition and behavior
call the red box a car            -> object-name alias
```

The CLI stores these rules, created entities, world state, and recent turns in
`~/.sieve/memory.json`. On startup the interpreter restores the aliases, applies
saved entity locations, and reloads the typed transcript. `knowledge` lists the
active rules, `history` lists turns, `reset` clears the physical world while
keeping learned vocabulary, and `forget learning` erases everything learned.

This is also what "training" means here. New behavior is added as explicit
rules and executable examples, never as gradient updates. The multi-turn test
suite is a conformance corpus: it fixes which continuations must resolve to
which grounded readings.

### Concept universe versus physical world

The lexical catalog is much larger than the grounded world. `ConceptUniverse`
exposes all bundled WordNet entries, including 57,384 noun concepts. A concept
becomes an `Entity` only when the user asks to materialize it:

```text
add a red elephant on the table
```

The new entity is persisted and can subsequently be moved, described, and
queried. Without materialization, `what is an elephant` answers from lexical
knowledge while `where is the elephant` correctly reports that no grounded
elephant exists. This keeps open-vocabulary breadth separate from the closed
world's executable constraints.

## 7. Zero survivors and repair

If the intersection is empty, the utterance is incompatible with the current
world or lexicon. SIEVE does not invent a reading. It reports the constraint
that failed.

Examples from the prototype:

```text
put the red cup in the table
-> The table is not a container.

put the moon on the table
-> I recognize 'moon' as English noun, but it is not bound in this world.
```

For an unrecognized or unbound noun, the prototype turns the failure into a
repair dialogue.
The user can bind the word to an existing kind:

```text
car is a box
```

or to a specific object:

```text
call the red box a car
```

A kind must narrow the noun. Generic kinds such as `thing`, `object`, `item`,
`entity`, `stuff`, `container`, or `location` are rejected: they would match
every non-room object and turn one request into a large artificial ambiguity.
The reply states the constraint and leaves the world unchanged. A kind binding
imports behavior already understood by the interpreter; it is not a claim that
every hat literally is a box.

The original utterance is then retried. This is explicit vocabulary extension,
not statistical learning: the new word receives a declared constraint, and the
system immediately uses it.

In a fuller system, the report would be computed from a minimal conflict set:
the smallest collection of lexical and world constraints whose removal would
make the utterance satisfiable. That turns failure into a precise repair
request rather than a generic "I didn't understand."

## 8. Execution as verification

For a command, the surviving reading contains an action with grounded arguments.
Execution is direct:

```text
put(theme=red_cup, destination=on(table))
-> red_cup.location := table
```

The action is not a separate semantic layer chosen after language
understanding. It is the reading. Understanding and execution are two views of
the same grounded interpretation.

## 9. Algorithm

```text
understand(text, world, context):
    text = apply_context_rules(text, context)
    draft = assemble(text)
    readings = ground(draft, world, context)
    readings = [r for r in readings if all_constraints_hold(r)]

    if readings is empty:
        return error(minimal_conflict(readings))
    if len(readings) == 1:
        result = execute_or_answer(readings[0])
        context = commit_context(context, readings[0], result)
        return result

    pending = readings
    q = choose_question(pending)
    return ask(q, pending)


answer(answer_text, pending, world, context):
    ref = assemble_reference(answer_text)
    selected = resolve(ref, pending.entities)
    survivors = [r for r in pending if r.entities intersects selected]

    if survivors is empty:
        return error("nothing matches")
    if len(survivors) == 1:
        return execute_or_answer(survivors[0])
    return ask(choose_question(survivors), survivors)
```

## 10. Why it is efficient

SIEVE spends computation only where ambiguity actually exists:

- no forward pass through a large parameter matrix;
- no decoding over a vocabulary;
- no training corpus;
- no similarity search over embeddings;
- short commands produce small candidate sets;
- questions cut the remaining set before the system acts.

The prototype's worst case is the product of candidate counts for noun phrases
and destinations. That is exponential in sentence length in theory, but the
system is lazy: it resolves only what the utterance needs and asks before
expanding the next ambiguous slot. A production version would replace the
Cartesian product with constraint propagation over a graph.

The important efficiency is not raw speed on a benchmark. It is **auditability**:
every reading, every elimination, and every question has a reason.

## 11. What would falsify the approach

The mechanism is only useful if a bounded lexicon plus grounded constraints can
cover enough language for a real task. It would fail if:

- most utterances require open-world knowledge that is not in the situation
  model;
- ambiguity is so pervasive that questioning becomes annoying;
- the constraint grammar becomes as large and opaque as the model it replaces;
- real users need paraphrase and metaphor more than grounded commands.

Those are empirical questions. The prototype makes them testable instead of
hiding them inside a model.

## 12. Honest scope

The individual ingredients - constraint satisfaction, frame semantics, active
questioning, situation models - are not new by themselves. The mechanism here is
the combination and the policy:

1. represent meaning as the surviving set, not a chosen reading;
2. treat syntax as another constraint source rather than a prior stage;
3. never guess when a discriminating question is available;
4. make execution the test of grounding;
5. treat unknown words as missing rules, not as material to hallucinate.

That is the invention this repository explores.
