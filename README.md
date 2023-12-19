This framework will consist of various tools and methods to make AI development, experimentation, and productionalizing easier.  

Some tools will cover:
- Chat Interfaces, takes care of the boilerplate code that is necessary to every chat app
- Monitoring, will include extensions of MLflow and custom logging and monitoring solutions
- Evaluating, will include extensions of MLflow but mostly custom tooling for prompt variation and model variation eval and comparison
- Fine tuning, will include both tools for generating fine tuning datasets as well as methods to start fine tuning with the datasets



Data Gen Pipeline:
- With a source of text, maybe a collection of PDFs
- Parse documents into sections
- Use a series of LLM prompts to generate summaries, questions, and answers for each section
- Iterate over your documents to create QA datasets with 100s-1000s of pairs specific to your documents
- Fine tune a model on those QA pairs to get better and up to date answers with the new and specific information
