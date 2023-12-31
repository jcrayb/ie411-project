# READ ME:
## Project structure:

There are multiple variations of the problem:
- __primal.py__ : This file contains the full primal problem, with no optimizations. It considers every pixel as part of its search space.

- __primal+dual.py__: This file contains the full primal problem, as well as the dual problem, and checks the complementary slackness at every iteration. As before, it also contains no optimizations.

- __optimized.py__: This file contains the primal problem only, and utilizes the node reduction techniques explored in the associated paper.

The repository also contains the "illustration" folder. This contains any extra scripts which were used to create explanatory images for the paper.

The final images can be found in the "final_images" folder. It contains both the output images from the full problem ("new_img.png") and the optimized problem ("optimized.png")

## Performance:

We compared the performance of the full and optimized problems on two different computers. One is a laptop with an Intel i5-10300H processor (plugged in, 100%, performance setting), and the other is a desktop with an AMD Ryzen 9 5900X. For each test, we used the average of 5 runs. Here are the results:

||Laptop|Desktop|
|---|---|---|
|Full problem|24.835 s|15.446 s|
|Optimized problem|10.948 s|6.447 s|

We can see that in both cases, there is around a 60% reduction in computing time, reflecting the halving in number of nodes.

## Final Result:

Here is the final image after 25 iterations:
<div style="display: block;
    margin:auto;
    width:min(500px, 100%);">
    <img src="https://files.jcrayb.com/files/ie411/project/new_img.png" 
alt="Final Image"
style="width:100%"
>
</div>