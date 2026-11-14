## Definition of Done (Project Rubric)
Every task and file created in this repository must strictly align with these required deliverables:
1. **Working Repository:** Runnable script, notebook, or modular codebase with clear dependencies (`requirements.txt` or `package.json`) and `.env.example`.
2. **5-Stage Architecture:** Explicit separation of concerns in both code and documentation:
   - **Input:** user query validation and normalization
   - **Knowledge:** context/data retrieval logic
   - **Tools:** structured tool/function definitions
   - **Decisions:** routing, tool-selection, and refusal logic
   - **Output:** formatted final response generation
3. **Boundary Enforcement:** Strict scope guardrails so the agent reliably holds its boundary and politely refuses out-of-scope requests.
4. **Example Interactions (`examples/interactions.md`):** Real transcripts generated from running the code, including standard queries and at least one query testing and holding its boundary.
5. **Comprehensive `README.md`:** Covers how to run, models used, libraries, custom logic, the 5-stage architecture, and example interactions.
6. **Optional Demo:** A lightweight frontend/CLI interface (`app.py`) or demo walkthrough script.
