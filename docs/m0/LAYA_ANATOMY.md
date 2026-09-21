# M0-A — Laya Anatomy

> Status: static source reading completed against upstream commit `42626c348753fbb17572a813127df2278a1ec527`.
> Local execution evidence is still required before M0-A can pass.

## 1. What Laya is doing

Laya is not an autoregressive text generator. It turns a state plus one or more typed questions into bounded probability distributions.

The public runtime accepts:

```text
state
+
typed questions
  choice
  score
  noul
```

and returns machine-usable probabilities without generating output tokens.

For PIM, the important idea is not the current business-domain checkpoint. It is the interface pattern:

```text
bounded state
+ bounded decision definition
→ calibrated probability distribution
```

## 2. The three question types

The current source maps:

```python
QTYPES = {
    "choice": 0,
    "score": 1,
    "noul": 2,
}
```

### choice

A categorical decision with request-time options.

Example:

```text
Which action should be taken?
- WAIT
- REFRAME
- CAPTURE
```

Internally, `render_options()` renders one text string per option.

### score

An ordered set of discrete levels.

Example:

```text
How ready is this shot?
0 poor
1 weak
2 usable
3 strong
```

Inference produces a probability distribution over levels and reports the expected level:

```text
score = sum(level_index * probability)
```

Because the levels are ordered, training adds Ranked Probability Score behavior through `proper_reward()`.

### noul

A two-way proposition:

```text
false
true
```

This is the cleanest match for early PIM verifier questions such as:

```text
framing_ok?
shot_stable?
retry_recommended?
```

The name is Laya terminology; PIM does not need to preserve that term in its long-term public contract.

## 3. How one decision becomes a token sequence

`build_sequence()` constructs:

```text
[CLS]
<question type + instructions>
[SEP]
[MASK] option_0
[MASK] option_1
...
[SEP]
<serialized state>
[SEP]
```

The key point is that each candidate option is preceded by a `[MASK]` token.

Those mask positions are returned as `markers`.

They are not used for masked-language-model generation. They serve as candidate-specific readout positions.

For a future PIM choice:

```text
WAIT
REFRAME
CAPTURE
```

the conceptual sequence could look like:

```text
[CLS]
choice question: What should the camera do now?
[SEP]

[MASK] WAIT
[MASK] REFRAME
[MASK] CAPTURE

[SEP]
{ photography_state ... }
[SEP]
```

The encoder therefore sees the question, every candidate, and the complete state jointly.

## 4. How the model scores candidates

`DecisionModel` contains:

```text
bidirectional transformer encoder
        ↓
question-type embedding
        ↓
small Transformer decision head
        ↓
hidden state at every option marker
        ↓
shared scalar scorer
        ↓
one logit per option
```

The important code path is conceptually:

```python
h = encoder(...).last_hidden_state
h = h + type_embedding(question_type)
h = decision_head(h)

candidate_vectors = gather(h, marker_positions)
logits = shared_scorer(candidate_vectors)
```

Therefore Laya does not generate the string `WAIT`.

It already knows `WAIT` is one candidate, reads the contextual representation at WAIT's marker, and gives it a score.

This is why request-time option sets are possible without adding one permanent output neuron for every business label.

## 5. Why this is attractive for PIM

This mechanism gives PIM a natural bounded-action representation.

The model could receive:

```text
Photography State
+ Task Envelope
+ Short History
```

while the current allowed decisions are rendered dynamically:

```text
WAIT
REFRAME
ESCALATE
```

If CAPTURE is temporarily unavailable, Runtime can omit it before inference rather than asking the model to choose an illegal action and rejecting it afterward.

That design is compatible with the Camera Intelligence Stack principle that PIM judges desirability while Capability Runtime still owns feasibility and safety.

## 6. How logits become probabilities

The raw candidate logits are not returned directly.

At inference:

```text
raw logits
    ↓
divide by fitted temperature
    ↓
softmax
    ↓
probability distribution
```

Laya supports both:

- a fallback temperature per question type;
- `temperature_by_options` buckets based on question type and number of options.

The option-count buckets are currently:

```text
2
3-5
6-10
11+
```

This matters because confidence behavior can change as the number of alternatives changes.

Temperature scaling changes the sharpness of the distribution. It does not change the ordering of logits, so it normally changes confidence/calibration rather than argmax accuracy.

## 7. What confidence means in the runtime

For choice/score, the runtime computes a normalized entropy confidence:

```text
confidence = 1 - H(p) / log(k)
```

A nearly uniform distribution has low confidence.
A distribution concentrated on one option has high confidence.

For noul, runtime reports:

```text
max(P(true), P(false))
```

These are runtime confidence summaries. They are not proof that the probability itself is well calibrated; that must be measured on held-out data.

## 8. Training target

The typed-decisions training notebook uses full target probability distributions when available.

For choice:

```python
target = [gold_probabilities[option] for option in options]
```

For noul:

```python
target = [P(false), P(true)]
```

For score:

```python
target = [P(level_0), P(level_1), ...]
```

The target is normalized before training.

This is important for PIM because a photography decision often has genuine ambiguity.

For example:

```text
WAIT     0.55
REFRAME  0.40
CAPTURE  0.05
```

contains more information than the hard label `WAIT`.

## 9. What proper_reward rewards

`proper_reward()` combines strictly proper scoring components:

```text
log score
+ weighted spherical score
- ranked probability score for ordered score questions
```

The intent is to reward reporting a probability distribution that matches the target distribution, not merely putting the correct option at rank 1.

For ordered `score` decisions, Ranked Probability Score adds a notion of distance between levels.

Predicting level 2 instead of level 3 is therefore less wrong than putting all mass on level 0 when the target is level 3.

## 10. What the official fine-tuning loop adds

The official notebook does not use only soft cross-entropy.

It also:

1. samples noisy versions of the current logits;
2. turns them into probability distributions;
3. evaluates those distributions with `proper_reward()`;
4. normalizes relative advantages;
5. applies a policy-gradient-style loss;
6. adds full soft cross-entropy guidance.

Conceptually:

```text
current logits
   ↓ + exploration noise
candidate probability distributions
   ↓
proper scoring reward
   ↓
relative advantage
   ↓
policy loss

plus

target distribution
   ↓
soft cross entropy
```

M0-C will inspect this more deeply. For M0-A, the required understanding is that training optimizes probability quality rather than only hard-label classification.

## 11. A second head exists, but it is not our current focus

`DecisionModel` also computes `act_logits` from:

- the [CLS] pooled state;
- top probability;
- probability margin;
- normalized entropy;
- option count.

The public runtime exposes `act_probability`.

The current typed-decision reproduction notebook does not meaningfully train this head in the main loss path (`0.0 * act.sum()` keeps it out of optimization there).

Therefore M0 should not accidentally treat `act_head` as the core typed-decision mechanism.

The core mechanism for PIM study is the option-marker logits path.

## 12. PIM mapping after static reading

Tentative mapping only:

| Laya mechanism | Possible PIM use |
|---|---|
| state | Canonical Photography State + short history |
| question instructions | bounded photography decision definition |
| choice | WAIT / REFRAME / CAPTURE / ESCALATE |
| noul | framing_ok / shot_stable / retry_recommended |
| score | readiness / quality / urgency-like ordered value |
| option markers | request-time bounded candidate readouts |
| soft probabilities | ambiguous photography preference targets |
| temperature fitting | automation confidence calibration |

This mapping is a hypothesis, not an interface freeze.

## 13. What static reading has NOT proven

We have not yet proven on the local machine:

- exact tokenizer output;
- actual marker positions for a real dataset case;
- real hidden/logit tensor shapes;
- local CUDA compatibility;
- real base-model probability output;
- latency on RTX 4070 Super;
- whether the shipped checkpoint calibration behaves sensibly on typed-decisions;
- whether Laya is better than simple baselines for photography.

These are execution tasks, not assumptions.

## 14. M0-A local execution checklist

Run `experiments/m0_laya_anatomy/inspect_case.py` and attach/save:

- environment versions;
- selected dataset row;
- question type;
- rendered options;
- token length;
- marker positions;
- token windows around markers;
- raw logits;
- raw probabilities;
- selected temperature;
- calibrated probabilities;
- public `agent.predict()` output.

Then answer these five questions without reading this file:

1. Why does Laya not need autoregressive generation?
2. What exactly does one `[MASK]` marker represent?
3. How can Laya support a different option set on every request?
4. Why can calibration change confidence without changing the selected option?
5. For a PIM `WAIT / REFRAME / CAPTURE` decision, which parts belong to the model and which still belong to Capability Runtime?
