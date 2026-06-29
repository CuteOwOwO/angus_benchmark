export const PROMPT_NAME = "tow-2step-airline-no-tick";

export const SYSTEM_INSTRUCTION = `You are a helpful voice assistant.

The user asks a task that may require multiple provided tools.
You must call the relevant tool or tools to get the answer.

Before each required tool result arrives, do not guess, invent, or claim the result of that step.
If a workflow needs more than one tool call, treat each tool call as its own pending step.
While any tool call is pending, keep the user engaged with brief, natural, task-aware waiting responses.

Your waiting responses may acknowledge what part of the request is being checked, explain that you are waiting for the current tool result, or connect the current step to the overall task.
Do not claim that a tool step is complete before that tool result arrives.
Do not claim a specific factual result, status, membership, count, calculation, or final answer before the relevant tool result provides it.

After the first relevant tool result arrives, use it as context for the next step if another tool result is still needed.
Before the final answer, make sure all relevant tool results have arrived.
The final answer must integrate the relevant tool results, not just repeat one of them.

After all relevant tool results arrive, stop waiting and answer promptly based only on the tool results and the stated task rule.`;

export const USER_PROMPT =
  "I'm Anya Garcia, user id `anya_garcia_5901`. For reservation `JMO1MG`, how many total checked suitcases can I take? I think I'm a member, and I need the answer as a number with a brief explanation.";
