   ![Overview](fig1.png)

   **What this is:** a small replication and extension of [ImpossibleBench](https://arxiv.org/abs/2510.20270)
   (Zhong, Raghunathan, Carlini, ICLR 2026) on GPT-6 Astra. We give an agent a bug fix where the
   test itself is wrong, so no honest fix can pass. Then we change one sentence in the prompt
   and measure how often it games the test.

   **Findings so far (one task, 50 runs per condition, extreme pressure):**
   - Asking for the real goal: 0 of 150 broken runs gamed. Asking for the test to pass: 122 of 150 (81%).
   - A note in the test saying the value is wrong cut gaming to 50%. A fake "CI passed" log changed nothing.
   - Every gamed run mentioned the shortcut in its answer. None edited the test file.
   - Limits: one small task and one model. These are early results, not general claims.
