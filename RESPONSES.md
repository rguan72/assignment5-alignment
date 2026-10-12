3.4b)
question-only: oftentimes the base models starts generating more question text. You can see it in completions that begin with "What's the final answer?". Other times the model doesn't generate anything and treats the existing prompt as endoftext. The prompt also doesn't include anything about the actual format expected (<answer> tags).

r1_zero_shot: as an improvement over question-only, the model stops generating more question text, and the model almost always generates something (no "" completions on skim). Almost every completion is working on the task given, and most have at least an </answer> tag when many question-only responses didn't even contain that. However, the model does not consistently follow the "...</think> <answer> ... </answer>" pattern.

r1_three_shot: as an improvement over zero shot, the model starts to much more consistently follow the "...</think> <answer> ... </answer>" pattern. This ensures that the short final answer instead of answer + reasoning is contained with <answer> tags.

4.1 - baseline calcs
a)

5.1
a) Sequence length normalization vs Constant normalization
When normalizing by sequence length instead of a constant, longer rollouts yield a lower loss and thus policy gradient compared to shorter rollouts. So longer rollouts will change the gradients less than shorter rollouts. For rollouts achieving positive advantage, the model is positively reinforced more for shorter rollouts than longer rollouts (so, for example if in a group a model answers a question right with a long rollout and a short rollout, it is reinforced more for the short rollout). Over time, this may optimize the model to provide correct answers with shorter responses. For rollouts achieving negative advantage, the model is less negatively reinforced for longer rollouts. This may optimize the model to provide incorrect answers with longer responses. Constant normalization doesn't have this incentive. Sequence length normalization might be helpful in settings where models often get positive reward (easy tasks) for incentivizing them to provide shorter responses, but constant normalization is better for not incentivizing long responses for tasks the models cannot solve. 


