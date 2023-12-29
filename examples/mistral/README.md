## Mistral Fine Tuning Examples

### Data Generation Pipeline
1. Starting with a source of information, in this case a collection of Arxiv URLs linked to papers written by various Matrix Lab faculty
2. Parse the information source into subsections or chunks, resulting in each paper turning into 20-40 chunks
3. Prompt a language model to generate a summary on the chunk, another prompt to generate a question relevant to the chunk, this case GPT-3.5-turbo
4. Prompt the model again with retrieval, to provide an answer to the question directly referencing the text
5. The goal is to refine both the parsing and prompting to generate as many QA pairs, incorporating as much of the information specifc to your document collection as possible

### Evaluation Pipeline
1. Using the Chunk:Summary:Question:Answer pairs generated above
2. Use the QA pairs generated with retrieval as your Ground Truth
3. Generate new Answers to your questions, with no retrieval, and the pre fine tuned version of your model, as your Baseline
4. Generate new Answers to your questions, with no retrieval, and the post fine tuned version of your model
5. Use whatever metric your want to compare the improvement from Baseline to Fine Tuned ie. similarity to ground truth, or maybe another prompting pipeline but with GPT-4 as the evaluator