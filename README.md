### IMPORTANT INFORMATION:

There are multiple variations of the problem:
- __primal.py__ : This file contains the full primal problem, with no optimizations. It considers every pixel as part of its search space.

- __primal+dual.py__: This file contains the full primal problem, as well as the dual problem, and checks the complementary slackness at every iteration. As before, it also contains no optimizations.

- __optimized.py__: This file contains the primal problem only, and utilizes the node reduction techniques explored in the associated paper.

The repository also contains the "illustration" folder. This contains any extra scripts which were used to create explanatory images for the paper.

Here is the final image after 25 iterations: \
<img 
src="https://files.jcrayb.com/files/ie411/project/new_img.png" 
alt="Final Image"
style="display: block;
    margin:auto;
    width:min(500px, 100%);">