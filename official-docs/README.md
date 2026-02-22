# Official Docs: LangChain Deep Agents

## Quickstart

I initially created `01_first_agent.py` for the first example, taken from official docs' [overview page](https://docs.langchain.com/oss/python/deepagents/overview), but it's not very useful so I removed it.

`02b_quickstart_stream.py` is funny. It's from quickstart page, [streaming section](https://docs.langchain.com/oss/python/deepagents/quickstart#streaming), which redirects you to [here](https://docs.langchain.com/oss/python/langchain/streaming/overview) which only work if we use `from langchain.agents import create_agent` instead of `from deepagents import create_deep_agent`. Yaa begitulah lika liku belajar framework langchain.

## Customization

- [v] Model: use `model` in `create_deep_agent()`
- [v] Tools: use `tools` in `create_deep_agent()`
- [v] System prompt: use `system_prompt` in `create_deep_agent()`
- [ ] Middleware -> use `middleware` in `create_deep_agent()`. Ada codenya, ga dikasih contoh outputnya.
    - many middlewares available, read each one
    - `handler`? no explanation about what this is?
- [v] Subagents: use `subagents` in `create_deep_agent()`
    - use either `SubAgent` (a dict) or `CompiledSubAgent` (an actual agent) instead
    - in the [Using CompiledSubAgent](https://docs.langchain.com/oss/python/deepagents/subagents#using-compiledsubagent) code, it uses `create_agent()` with a param `prompt`, while `langchain.agents.create_agent` don't have this param. Should be `system_prompt`. This docs is testing my patience.
- [v] Backends: use `backend` in `create_deep_agent()`
    - the implementation code dont give good example of use case prompt and tools
- [ ] Sandboxes: use Daytona or Deno, put into `backend` in `create_deep_agent()`. Gonna try
- [ ] Human-in-the-loop: use `interrupt_on` and `checkpointer` in `create_deep_agent()`. Gonna try
- [ ] Skills: SKILL.md into `create_deep_agent()`
- [ ] Memory: AGENT.md into `create_deep_agent()`
- [v] Structured ouput: `response_format` in `create_deep_agent()`
- [ ] MCP?

## Backend

Special section because this is I think the main feature of deep agents

### StateBackend (ephemeral)
already there?

### FilesystemBackend (local disk)
grants agents direct filesystem read/write access. scary

### LocalShellBackend
the local shell might not have the runtime needed or packages installed. i think i'll use sandboxes instead