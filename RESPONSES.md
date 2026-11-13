3.4b)
question-only: oftentimes the base models starts generating more question text. You can see it in completions that begin with "What's the final answer?". Other times the model doesn't generate anything and treats the existing prompt as endoftext. The prompt also doesn't include anything about the actual format expected (<answer> tags).

r1_zero_shot: as an improvement over question-only, the model stops generating more question text, and the model almost always generates something (no "" completions on skim). Almost every completion is working on the task given, and most have at least an </answer> tag when many question-only responses didn't even contain that. However, the model does not consistently follow the "...</think> <answer> ... </answer>" pattern.

r1_three_shot: as an improvement over zero shot, the model starts to much more consistently follow the "...</think> <answer> ... </answer>" pattern. This ensures that the short final answer instead of answer + reasoning is contained with <answer> tags.

4.1 - baseline calcs (in notebook)
a)

4.3c) I ran this experiment with seeds 0, 1, 2, 3.

Before training rollouts:
  Let’s break it down step by step. First, we know that a full can of frosting can frost a layer cake, single cake, or a dozen cupcakes. 
  
  For the layer cakes, since three layer cakes are needed and each is a layer cake, and each layer cake requires a can of frosting, the total cans needed for layer cakes is 3 * 1 = 3 cans. 
  
  For the single cakes, since there are 6 single cakes and each can of frosting can frost one cake, the total cans needed for single cakes is 6 * 1 = 6 cans. 
  
  For the dozen cupcakes, since a half can of frosting is used to frost a single cake or a dozen cupcakes, and each can of frosting can frost one cake or a dozen cupcakes, the total cans needed for a dozen cupcakes is 0.5 * 1 = 0.5 cans. 
  
  Now, adding all the cans needed for each type of dessert, we get 3 + 6 + 0.5 = 9.5 cans. However, the question asks for the total cans needed and not the total frosting amount. So, the total cans needed is 9.5, but since we can’t have half a can, we round up to 10 cans. Therefore, Paul will need 10 cans of frosting to have everything ready and frosted for customer pick up. </think>

  Each bird eats 7 berries a day. Samuel has 5 birds. 5 birds * 7 berries/ bird/day = 35 berries/day. Samuel's birds eat 35 berries in 1 day. </think> Each bird will eat 35 berries in 4 days. 35 berries/day * 4 days = 140 berries. Samuel's birds will eat 140 berries in 4 days. </answer>

After training rollouts:
 First, we find out how many students enrolled in contemporary dance: 20 * 0.2 = 4 students. Then, we find out how many students are left: 20 - 4 = 16 students. Next, we calculate how many of these remaining students enrolled in jazz dance: 16 * 0.25 = 4 students. So, the rest, which is the number of students who enrolled in hip-hop dance, is 16 - 4 = 12 students. Therefore, 12 students pursued hip-hop dance. The percentage of students pursuing hip-hop dance is (12 / 20) * 100 = 60%. </think> <answer> 60 </answer>

  James runs 3 sprints, and each sprint is 60 meters. So, in one session (3 sprints), he runs 3 * 60 = 180 meters. Since he does this 3 times a week, he runs 180 * 3 = 540 meters in a week. </think> <answer> 540 </answer>

4.3 - Problem (grpo_learning_rate): Tune the learning rate (4 B200 hrs) (3 points)
uv run modal run experiments/grpo_experiments_standard_on_policy.py --learning-rate 1e-6
uv run modal run experiments/grpo_experiments_standard_on_policy.py --learning-rate 1e-4

[lr-ablation.png](lr-ablation.png)
Both learning rates performed significantly worse than 1e-5. The higher learning rate maxed out format reward then barely learned after that. The lower learning rate increased at a very slow rate.

4.3 - Problem (grpo_prompt_ablation): Prompt ablation (4 B200 hrs) (3 points)
uv run modal run experiments/grpo_experiments_standard_on_policy.py --prompt-fname "cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt"
uv run modal run experiments/grpo_experiments_standard_on_policy.py --prompt-fname "cs336_alignment/prompts/question_only.prompt"

[prompt-ablation.png](prompt-ablation.png)

Three shot prompt starts off with a higher val and format reward than r1 zero shot prompt and generally has a higher val reward per step. question only and r1 zero shot both start with zero val reward, but quesiton only starts off with zero format reward and takes many steps to catch up to the approximate val reward of r1 zero. Question only requires many iterations to even learn the question-only nature of the task.  

5.1
a) Sequence length normalization vs Constant normalization
When normalizing by sequence length instead of a constant, longer rollouts yield a lower loss and thus policy gradient compared to shorter rollouts. So longer rollouts will change the gradients less than shorter rollouts. For rollouts achieving positive advantage, the model is positively reinforced more for shorter rollouts than longer rollouts (so, for example if in a group a model answers a question right with a long rollout and a short rollout, it is reinforced more for the short rollout). Over time, this may optimize the model to provide correct answers with shorter responses. For rollouts achieving negative advantage, the model is less negatively reinforced for longer rollouts. This may optimize the model to provide incorrect answers with longer responses. Constant normalization doesn't have this incentive. Sequence length normalization might be helpful in settings where models often get positive reward (easy tasks) for incentivizing them to provide shorter responses, but constant normalization is better for not incentivizing long responses for tasks the models cannot solve. 

5.2 - think about RFT)



