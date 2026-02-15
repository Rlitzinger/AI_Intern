
[MD cheatsheet](https://www.markdownguide.org/cheat-sheet/)
# Dev Log

## 2/2/26 Started the AI Intern
- been working on system design for about 3 weeks now in local-agent-lab
- decided to go with a new project testing out claude's capabilities
- so far Claude has gone above and beyond chatgpt. It's clear, brief, provides great critical feedback, and gives a good perspective on my system design
    - in particular helped with dianosing my biggest issue: huge scope on the planning agent
- I am using this markdown file as a dev log to hopefully post content from and keep my thoughts together
- My goals:
    1. Learn the best architectures for agentic systems.
    2. Creatively design systems that enable smaller LLMs to hit well above their weight class.
    3. Post content for engagement on linkedIn, showing that i'm a capable agentic AI system developer.
    4. Make something that I can help manage parts of my life with.
- Make the schemas manually at first to help you learn and understand what's going on
    - there's not point in doing this if you don't learn and show that you are capable.

### Markdown best practices 
    1. Keep it concise and scanable
    2. be directive, not descriptive
        - 'Use pep8 for classes' is better than 'we have been using pep8'
    3. Explicitly mention your stack
    4. Prune regularly
    5. Include a project overview.
    6. Mention key architecutre, file structure

- made a claude.md file, learned the structure and best practices, filled it in with basic stuff

## 2/3-5/2026 Studying AI system Design
- did some research on Ai systems to see what would be the best to use
    - there's levels to this
1. - voting: ask multiple agents the same qeustion, take the majority answer
    - leader-follower: designate one agent as the authority, others defer to it.
    - Round-robin: one agent proposes an action, others check it before execution
    - !! These are simple. use them only for basic fact-checking, redundancy for reliability
2. - debate and critique: agents aregue for different positions, a judge evaluates teh arguments.
    - proposal-review cycles: one agent drafts, another critiques, the first revises
    - specialized roles w/ checks: a planner proposes, a critic finds flaws, an executor acts, a verifier confirms
    - confidence-weighted aggregation: agents report not just answer but confidence levels. Final decisions weight more confidence agents more heavily
    - hierarchial consensus: low level agents reache local consensus which feeds up to higher-level agents who sysnthesize across groups
    - !! Challenges at level 2 are handling disagrement that isn't resolved by discussion!!
    - !! How do you prevent agents from being too agreeable??
    - !! How do you bound the deliberation time
3. - Byzantine fault tolerance BFT: assumes nodes either work correctly or fail. Protocol handles this soft failure with partial trust and probabilistic correctness
    - mechanism design for agents: use game-theory principles to design rules where truthful cooperative behavior is the optimal strategy for each agent, even if self interseted in competing objectives
    - verifiable computation: agents provide not just answer but proofs or traces for others to check
    - constitutional approaches: agents are given explicit principles or constitutions. Consensus is reached by reference to shared principles rather than pure majority rules. Disagreements are resolved by appealing to the hierarchy of principles
    - Emergent consensus in swarms: with many argents the local interaction rules lead to global convergence (ants, birds, beeds) without central coordination.
    - meta-consensus: agents not only agree on answer but on how to agree - selecting the consnsus mechanism that is based on problem type

## 2/6/2026
- working today on the schemas for the system

## 2/12/2026 
- didn't really log the last few days, but worked on the schemas some more and refined them down
- made some functions to serialize the schema, save, and retrieve it
- using pydantic to make sure the schemas are enforced, rather than just placeholders
- found out what the -> after a function does, apparently it's just meta data??
- created the ollama call and was able to generate a plan from that (and track tokens etc.)
