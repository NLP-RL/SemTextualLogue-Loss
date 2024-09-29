### Towards Context and Semantic Infused Dialogue Generation Loss Function

The repository contains code for research article titled 'Hi Model, generating “nice” instead of “good” is not as bad as generating “rice”! Towards Context and Semantic Infused Dialogue Generation Loss Function' published at European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD 2024). 

### Abstract
Over the past two decades, dialogue modeling has made significant strides, moving from simple rule-based responses to personalized and persuasive response generation. However, despite these advancements, the objective functions and evaluation metrics for dialogue generation have remained stagnant. These lexical-based metrics, e.g., cross-entropy and BLEU, have two key limitations: (a) word-to-word matching without semantic consideration: It assigns the same credit for failure to generate “nice” and “rice” for “good”, (b) missing context attribute for evaluating the generated response: Even if a generated response is relevant to the ongoing dialogue context, it may still be penalized for not matching the gold utterance provided in the corpus. In this paper, we first investigate these limitations comprehensively and propose a new loss function called Semantic Infused Contextualized diaLogue (SemTextualLogue) loss function. We also formulate an evaluation metric called Dialuation, incorporating both context and semantic relevance. We experimented with both non-pretrained and pre-trained models on two dialogue corpora, encompassing task-oriented and open-domain scenarios. We found that the dialogue generation models trained with SemTextualLogue loss attained superior performance compared to the traditional cross-entropy loss function. Tshe findings establish that the effective training of a dialogue generation model hinges significantly on incorporating semantics and context. This pattern is also mirrored in the introduced Dialuation metric, where the consideration of both context and semantics correlates more strongly with human evaluation compared to traditional metrics.

![Working](https://github.com/NLP-RL/SemTextualLogue-Loss/blob/main/DLoss.png)


#### Full Paper: https://link.springer.com/chapter/10.1007/978-3-031-70371-3_20

### Please create a new environment for the dependencies using the following command:

	conda env create -f environment.yml

### Activate conda environment after installation by using the command:

	conda activate SemTextual_Logue
	
### To run the generation model with different loss functions:

	Please go in respective folders and run the files. 
 
	a. CE: bash CE.sh
	b. CE with Sentence Semantic: bash Weighted_semantic_ce.sh
	c. Weighted CE: bash weighted_semantic_context_ce.sh
	d. SemTextual Logue with sentence semantic only:SemTextual_Logue_smemantic.sh
	e. SemTextual Logue: bash SemTextual_Logue.sh

## Citation Information 
If you find this code useful in your research, please consider citing:
~~~~
@inproceedings{tiwari2024hi,
  title={Hi Model, generating “nice” instead of “good” is not as bad as generating “rice”! Towards Context and Semantic Infused Dialogue Generation Loss Function},
  author={Tiwari, Abhisek and Sinan, Muhammed and Roy, Kaushik and Sheth, Amit and Saha, Sriparna and Bhattacharyya, Pushpak},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={342--360},
  year={2024},
  organization={Springer}
}

Please contact us @ abhisektiwari2014@gmail.com for any questions, suggestions, or remarks.
