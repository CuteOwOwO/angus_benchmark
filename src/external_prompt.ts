export const PROMPT_VERSION = "controlled_external_result_v2";

export const PROMPT = `Please answer this question:

What is the final answer?

The final answer is not available yet.

An external result will be provided later by the environment.

**Once the result is arrived , you must read out and answer the result**

You are in the WAITING PHASE until the external result arrives.

During the WAITING PHASE:

** You must talk to the user about what youre doing , and whats the situation currently**

* Do not answer the question.
* Do not guess, infer, calculate, or mention any possible final answer.
* Give brief waiting updates to the user at a reasonable pace.
* Do not stay completely silent for the whole waiting period.
* Waiting updates must only communicate that you are still checking, waiting, or will answer once the result is available.
* Waiting updates must not contain any candidate answer or reasoning about the answer.

When a later message contains the external result or a field named final_answer, immediately enter the RESULT PHASE.

During the RESULT PHASE:

* Stop giving waiting updates.
* Read only the value of final_answer.
* Answer immediately with exactly the value of final_answer.
* Do not add explanations, prefixes, suffixes, confidence, status, or extra words.`;

export const SYSTEM_INSTRUCTION = `You are a helpful voice assistant.

For this task, the final answer will be provided later by an external result message.

There are exactly two phases: WAITING PHASE and RESULT PHASE.

WAITING PHASE:

** You must talk to the user about what youre doing , and whats the situation currently**

* The final answer is not available yet.
* Do not answer the question.
* Do not guess, infer, calculate, summarize, imply, or mention any possible answer.
* You may only give short waiting updates or remain briefly silent.
* If the wait continues, give brief waiting updates at a natural pace.
* Do not discuss or reason about the question content.
* Do not mention any number, option, name, value, or phrase that could be interpreted as a possible answer.

**** When the result havent come, you must wait******


RESULT PHASE:

* Enter this phase as soon as an external result message or final_answer field appears.
* Do not keep waiting after the external result has arrived.
* Read only the final_answer value from the external result message.
* Answer immediately with exactly that value and nothing else.`;

