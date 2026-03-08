
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

# 2/15/2026
- designed the coding agent today
- it's able to take in a task from the plan and make code
- also updated the planning agent to not dissect simple tasks into like 5 steps
    - for ex. reading the number of rows from a .csv was loading in pandas, then a helper function to load the table, then another function to read ...
    - now it is a little more refined to just one step for this stuf
- important note for later: if the function brings in some libraries make sure that they are installed via pip, or give instructions not to use 3rd party libs
- You have a minimum verticle slice!!! you can write a plan, generate a plan with tasks, and generate code against those tasks!!
- added a validation agent that checks out if the code agent returned good code
- stripped example usage from the code agent output mechanically bc it wouldnt follow directions well. 

# 2/16/2026
- excited to code today, was thinking of more multi-level hierarchies and a mechanism to determine how deep into hierarchial decomp is needed, or when to just one-shot requests if they are simple enough
- work today is to make a better validator (compiles code, improves quality, maake sure it works), then an orchestration for error recovery and retry, then a task router.
- added in a better validator that runs a test over the code to make sure it works as intended, improved this, now it runs multiple self-generated tests and stores the output in the db
- about to work on orchestrator > should replace the whole main.py calls and run through all of it itself.
    - orchestrator works! able to retry errors as well on the coding agent if test don't pass.
- created task router that delegates based on keywords
    - able to identify which agent to pass info to
-  working on updating the validation engine to be able to spin up local server code to test (like flask django)
    - having issues with the server not spinning up and there being bad tests
- was able to create a harnass that executes servers, and sends/receives info from them
- starting work on reading in user data from a local folder in the project, and adding output folder to put info into from the project

# 2/22/2026
- tons stuff done today
- multi-step planner that uses the hierarchial decomp to simplify tasks down
    - looks at a plan (research x, code y, return file z)
    - if the task is complex (not simple) then it breaks it down further into more steps
    - if it's ambiguous, it tries to clarify the request (not clear)
    - so clear/ambiguous and simple/complex
    - makes branches if it's complex, and labels a leaf if it's clear and simple
- added context send between steps in a process
    - the coding agent will receive the output of the research agent
- file agent can now save python files too
- did a ton of work to make the process smoother, fixed a lot of error issues
- reorganized the file into sub-agent folders, based around each section
    - like one for planning, execution, orchestration, etc

# 2/23/2026
- claude did a massive rewrite of a ton of code and made a ton of changes overnight
- completely new settings, logging system, config is now pydantic, richer schemas, better request processing (avg became average)
- added anlaysis agent
- smarter task routing
- full 3 hierarchial planner
- orchestrator can preprocess prompts, works with dependencies, synthesizes answers so i can understand them, puts error history in context, resets goals on retry
- better guards against hallucinations

# 3/2/2026
- did a lot of work to upgrade to a tier 2 planning stage. 
- basically better at reading specs and runing through them 
- error classifier, early aborrt on unrecoverable errors (timeout, import issues)
- available libary awareness
- structure retry prmpting with category-aware correction prompts
- kv cache values passed to coding agent

# 3/3/2026
- pain in the ass day. moved away from ollama to llama.cpp and outlines
- with the new packages i can have more fine tuned control over the llm
- can used constrained decoding and use outlines to pass strict syntax in the llm
- honestly the hardest part so far was this. just getting the system in check

# 3/4/2026
- did a lot more work on the planner
- added in constrained decoding for planning output to pick specific agents to use
- added a better classification of the planner
- provided context to the planner on what agents are available
- made a clear part of the plan to focus on exact user reqs (we were just making the same CRUD app without specifics over and over)
- added a critiqueagent and output contract
- updated the plan.py to better show details of what happened during the planning phase, and the thinking/reasoning
- added a round of critique with a red team/blue team

# 3/6?/2026
- Update planning phase to have clear specs and output contracts at each step
- 