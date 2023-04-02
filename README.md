# Cutoff for ComfyUI

![screenshot of workflow](https://github.com/BlenderNeko/ComfyUI_Cutoff/blob/master/examples/screenshot.png)

### what is cutoff?

cutoff is a script/extension for the Automatic1111 webui that lets users limit the effect certain attributes have on specified subsets of the prompt. I.e. when the prompt is `"a cute girl, white shirt with green tie, red shoes, blue hair, yellow eyes, pink skirt"`, cutoff lets you specify that the word blue belongs to the hair and not the shoes, and green tot he tie and not the skirt, etc.

### how doe this work?

When you provide stable diffusion with some text, that text gets tokenized and CLIP creates a vector (embedding) for each token in the text. So if we have a prompt containing "blue hair, yellow eyes" some of the vectors coming out of CLIP will correspond to the "blue hair" part, and some to the "yellow eyes". When CLIP does this it tries to take the context of the entire sentence in mind. Unfortunately CLIP isn't always as great at figuring out that the "blue" in "blue hair" should really only modify the noun "hair" and not the noun "eyes" a bit further in the sentence.

So how do we deal with this? we can mask out the tokens corresponding to "blue" and ask CLIP to create another embedding. In this new embedding we have a set of vectors corresponding to "yellow eyes" that are not affected by "blue", because blue wasn't part of the tokens. If we then take the difference between our original vectors and  these new vectors we now have a direction we can travel in for the eyes to become more affected by "yellow" and less by "blue". If we do this for all the color relations in text we can travel to an embedding where each of these relations are more isolated.

### ComfyUI nodes
To achieve all of this, the following 3 nodes are introduced:

**CLIPRegionsBasePrompt:** this node takes the full original prompt

**CLIPSetRegion:** this node sets a "region" of influence for specific target words, and comes with the following inputs:
- region\_text: defines the set of tokens that the target words should affect, this should be a part of the original prompt.
- target\_text: defines the set of tokens that will be masked off (i.e. the tokens we wish to limit to the region) this is a space separated list of words. If you want to match a sequence of words use underscores instead of spaces, i.e. "the\_target\_word". If you want to match a word that actually contains underscores escape the underscore, i.e. "the\\_target\\_word"
- weight: how far to travel in the direction of the isolated vector 

**CLIPRegionToConditioning:** this node converts the base prompt and regions into an actual conditioning to be used in the rest of ComfyUI, and comes with the following inputs:
- mask\_token: the token to be used for masking. If left blank it will default to the <endoftext> token. If the string converts to multiple tokens it will give a warning in the console and only use the first token in the list.
- strict_mask: When 0.0 the specified target tokens will not affect the other specified areas but do affect anything outside of those areas. When set to 1.0 the specified target tokens will only affect their own region.
- start\_from\_masked: When 0.0 the starting point to travel from is the original prompt. When set to 1.0 the starting point to travel from is the completely masked off prompt. Note that specifically when all region weights are 1.0 there is no difference between the two

You can find these nodes under `conditioning>cutoff`

Finally, Here are some example images that you can load into ComfyUI:

![first example generation of a cute girl, white shirt with green tie, red shoes, blue hair, yellow eyes, pink skirt using cutoff](https://github.com/BlenderNeko/ComfyUI_Cutoff/blob/master/examples/ComfyUI_00671_.png)
![first example generation of a cute girl, white shirt with green tie, red shoes, blue hair, yellow eyes, pink skirt using cutoff](https://github.com/BlenderNeko/ComfyUI_Cutoff/blob/master/examples/ComfyUI_00672_.png)
![first example generation of a cute girl, white shirt with green tie, red shoes, blue hair, yellow eyes, pink skirt using cutoff](https://github.com/BlenderNeko/ComfyUI_Cutoff/blob/master/examples/ComfyUI_00673_.png)
![first example generation of a cute girl, white shirt with green tie, red shoes, blue hair, yellow eyes, pink skirt using cutoff](https://github.com/BlenderNeko/ComfyUI_Cutoff/blob/master/examples/ComfyUI_00674_.png)