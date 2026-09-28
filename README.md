# CI Triage Agent

Agentic CI triage bot that reads failing logs, reproduces errors, and opens fix PRs for approval.

**Domain:** Agentic AI
**Language:** python
**Demonstrates:** You can embed agents into engineering workflows.

## 7-day build plan

- [ ] Day 1: Scaffold the Python project with a mock CI pipeline simulator that generates realistic failing build logs (syntax errors, test failures, import errors) stored as local JSON/text files, plus a basic CLI entry point and pytest setup.
- [ ] Day 2: Build the log ingestion and parsing layer that reads mock CI log files, classifies failure types (test failure, lint error, build error, etc.), and extracts structured error metadata (file, line, message, stack trace) into a dataclass model.
- [ ] Day 3: Implement the error reproduction engine that takes parsed failure metadata and attempts to reproduce the error locally in a sandboxed subprocess, capturing stdout/stderr and confirming the failure matches the original log before proceeding.
- [ ] Day 4: Integrate an LLM-backed fix-suggestion agent (using the openai library with a stubbed/mock client when OPENAI_API_KEY is absent) that receives the structured error and reproduction output, then proposes a code patch in unified diff format.
- [ ] Day 5: Build the mock Git/PR layer that applies the proposed patch to a local test repo clone, runs the test suite against the patched code, generates a mock pull request object with branch name, diff, and test results, and persists it as a JSON artifact.
- [ ] Day 6: Add the human-in-the-loop approval workflow: print a rich summary of the proposed PR, prompt the user for explicit confirmation before 'merging' (writing final patch to disk), support reject/edit/approve commands, and log all decisions to an audit trail file.
- [ ] Day 7: Wire all components into a full end-to-end agent loop with a configurable orchestrator, add integration tests covering the happy path and edge cases (unfixable errors, user rejection, LLM mock fallback), and package the project with pyproject.toml and a sample .env.example.

_A comprehensive README with an architecture diagram is generated on Day 7._
