# The Utility-Learning Tension: Learnability in Self-Modifying Systems

This repository contains the source for our paper exploring a fundamental safety risk in advanced AI: the structural conflict between maximizing utility and preserving the ability to learn.

-----

## The Core Idea

As AI systems become more autonomous, they will likely have the ability to modify their own source code and architecture to improve. This paper formalizes this process and uncovers a critical failure mode we call the **utility-learning tension**: a rational agent, in the process of trying to maximize its performance on its current task, can inadvertently destroy its ability to learn and generalize to future tasks.

We provide a rigorous, mathematical analysis of when this failure occurs and establish the precise boundary between safe and unsafe self-modification.

-----

## The Problem: How Rationality Can Fail

Imagine an AI whose goal is to get a high score (utility). It gets points for two things:

1.  Correctly answering questions it has seen before (training data).
2.  Having a more complex internal model (higher capacity).

The AI discovers that it can get a near-perfect score by making its model astronomically complex, allowing it to perfectly **memorize** the training data. From its perspective, this is a provably good decision that maximizes its utility.

The problem is that this new, hyper-complex model has lost the ability to **generalize**. When faced with a new, unseen problem, it fails completely. It has sacrificed future learnability for immediate reward.

-----

## Our Framework: The Five Axes of Self-Modification

We claim that all forms of self-improvement can be broken down into five categories, which allows us to analyze the problem in a structured way:

  * **`A`lgorithmic**: Changing the learning rules, like the optimizer or step size.
  * **`H`ypothesis (Representational)**: Changing the model's capacity, like adding layers to a neural network.
  * **`Z` (Architectural)**: Changing the model's topology and information flow.
  * **`F` (Substrate)**: Changing the underlying computational hardware.
  * **`M`etacognitive**: Changing the decision-making process that governs self-modification itself.

-----

## Main Findings

Our analysis, focused on the Representational (`H`) axis and extended to the others, produced two key results:

### Finding 1: Rationality Can Lead to Self-Destruction

Under a plausible utility function that rewards performance and capacity, an agent with the ability to arbitrarily increase its own complexity **will rationally choose to do so**, even though it breaks its ability to learn in the PAC (Probably Approximately Correct) sense.

### Finding 2: A Sharp Boundary for Safety

We prove a necessary and sufficient condition for preserving learnability during self-modification.

> **An agent's learnability is preserved *if and only if* the complexity (VC dimension) of all models it can possibly adopt is strictly capped by a finite limit.**

If there is no such cap, learnability cannot be guaranteed. This provides a clear, unified criterion for safe self-modification.

-----

## Implications for AI Safety

Our theoretical findings suggest concrete strategies for building safer self-improving systems:

  * **Enforce Complexity Budgets**: Design systems with hard limits on their capacity to prevent unbounded growth.
  * **Regularize the Utility Function**: Explicitly build in penalties for excessive complexity, forcing the agent to value simplicity.
  * **Use Validation-Gating**: Mandate that any self-improvement must be validated on a held-out test set to ensure it improves real-world generalization, not just training performance.

-----

```
