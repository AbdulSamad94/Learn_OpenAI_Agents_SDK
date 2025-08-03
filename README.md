# Learn OpenAI Agents SDK

This is my personal learning repository for the **OpenAI Agents SDK** — a framework for building intelligent, multi-agent AI applications with guardrails, tools, orchestration, and observability.

## 📌 About
The OpenAI Agents SDK is a lightweight and flexible Python framework designed to make building agentic AI workflows easier. It supports:
- Single-agent and multi-agent setups
- Input and output guardrails for safety and correctness
- Tool integration and handoff between agents
- Built-in tracing and observability
- Structured output and type-safe responses

## 🧠 Key Concepts
- **Agent** – An intelligent unit that can reason, use tools, and produce structured outputs.
- **Runner.run()** – Executes an agent loop until an exit condition (tool call, structured output, error, or max turns).
- **Guardrails** – Validation layers for safety:
  - **Input Guardrails** – Filter/validate incoming user input before processing.
  - **Output Guardrails** – Validate final responses before returning to the user.
- **Tools / Handoff** – Agents can call tools or hand off tasks to other agents for multi-agent workflows.
- **Tracing & Observability** – Built-in tracing to debug and monitor agent behavior.


## 🚀 Getting Started
1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/learn-openai-agents-sdk.git
   cd learn-openai-agents-sdk

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate     # Mac/Linux
   venv\Scripts\activate        # Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run an example:

   ```bash
   python basics/simple_agent.py
   ```

## 🎯 Learning Goals

* Understand how to create and configure agents
* Implement input/output guardrails for safety
* Build multi-agent workflows using tools and handoff
* Use tracing for debugging and performance monitoring
* Develop real-world use cases like document analyzers and AI assistants

## 🛠 Useful Patterns

* **Agent as Tool** – One agent acting as a tool for another
* **Guardrail-based Filtering** – Validating and filtering before execution
* **Context Propagation** – Passing data between agents using `RunContextWrapper`

## 📚 References

* [OpenAI Agents SDK Docs](https://openai.github.io/openai-agents-python/)
* [Guardrails Documentation](https://platform.openai.com/docs/assistants/guardrails)
* [FastAPI + Agents SDK Integration Examples](https://fastapi.tiangolo.com)
* [Community Discussions & Examples](https://community.openai.com)

## 📝 Notes

* The SDK is updated frequently — check the official docs for the latest changes.
* TypeScript support and real-time/voice agent features are coming soon.

