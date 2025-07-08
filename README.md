# min-eval-framework

-----

# How to Evaluate Your RAG System: A Simple Guide with Python and AutoEvals

Retrieval-Augmented Generation (RAG) is a powerful technique that enhances Large Language Models (LLMs) by providing them with relevant, external information. But as with any powerful tool, you need to ensure it's working correctly. How do you know if your RAG system is providing accurate, relevant, and faithful answers? The answer is evaluation.

This blog post will guide you through a single-file Python script that demonstrates a simplified, yet robust, RAG evaluation process. We'll build a self-contained evaluator that you can easily adapt for your own projects to measure the quality of your RAG pipeline.

### Why Is RAG Evaluation So Important?

A typical RAG system has two core components: a **retriever** that fetches relevant documents (the context) and a **generator** (the LLM) that synthesizes an answer based on that context. Failure can happen at either stage:

  * **Irrelevant Context:** The retriever might pull documents that don't help answer the user's question.
  * **Unfaithful Generation:** The LLM might ignore the provided context and "hallucinate" an incorrect answer.
  * **Incomplete Answers:** The LLM might fail to synthesize all the relevant facts from the context.

To build trust and reliability in your application, you need to measure and mitigate these failure points.

### Our Toolkit

We'll use a few key Python libraries to build our evaluator:

  * **OpenAI:** To connect to an LLM for generating answers and running evaluations. We'll use the `openai` library with Azure, but it works just as well with standard OpenAI.
  * **Python-Dotenv:** To securely manage our API keys.
  * **AutoEvals:** A fantastic open-source library for running LLM-based evaluations. It provides pre-built metrics designed specifically for tasks like RAG.

## Setting Up Your Environment

Before we dive into the code, let's get your environment ready.

**1. Install the required libraries:**

```bash
pip install openai python-dotenv autoevals
```

**2. Create a `.env` file:**
In the same directory as your script, create a file named `.env`. This is where you'll store your credentials. **Never commit this file to version control.**

```ini
AZURE_OPENAI_ENDPOINT="your_endpoint_here"
AZURE_OPENAI_API_KEY="your_api_key_here"
```

**3. Run the script:**
Once the code is ready, you can run it from your terminal:

```bash
python demo.py
```

## Dissecting the Code: The `SimpleRAGEvaluator`

We've structured our entire logic inside a class called `SimpleRAGEvaluator`. This makes the code clean, reusable, and easier to understand. Let's walk through it method by method.

### Initialization (`__init__`)

The constructor sets up the core components: logging, configuration, and client initialization.

  * **Logging:** We set up basic logging to see what the evaluator is doing at each step.
  * **Configuration:** We define which models to use for generation (`gpt-4o-mini`) and evaluation (`gpt-4o-mini`). You can use different models for each task.
  * **Initialization:** It immediately calls helper methods to configure the OpenAI client and set up our evaluation metrics.

### Connecting to the LLM (`_initialize_clients`)

This method handles the connection to Azure OpenAI. It loads the credentials from the `.env` file, initializes the `AzureOpenAI` client, and runs a quick test to ensure the connection is successful. Robust error handling ensures the script will exit gracefully if credentials are not found.

### Defining Our Metrics (`_initialize_evaluators`)

This is where we define *how* we'll measure performance. We use a combination of pre-built metrics from `autoevals` and a simple custom one.

```python
# From the SimpleRAGEvaluator class
def _initialize_evaluators(self):
    """Initializes the autoevals library and defines metrics."""
    self.logger.info("Initializing evaluation metrics...")
    init(client=self.client) # Sets a global client for all evaluators
    
    self.METRICS = {
        'Context Relevancy': ContextRelevancy(model=self.evaluation_model),
        'Faithfulness': Faithfulness(model=self.evaluation_model),
        'Factuality': Factuality(model=self.evaluation_model),
        'Exact Match': self._exact_match
    }
    self.logger.info("Core evaluation metrics are ready.")
```

Here’s a breakdown of our chosen metrics:

  * **`ContextRelevancy`**: Measures if the retrieved context is relevant to the user's input prompt. A low score here indicates a problem with your retrieval step.
  * **`Faithfulness`**: Checks if the generated answer is grounded in the provided context. A low score means the LLM is hallucinating or adding outside information. This is critical for RAG.
  * **`Factuality`**: Compares the generated answer to a "ground truth" or expected answer to determine if it is factually correct.
  * **`_exact_match`**: A simple, custom function we wrote that checks for a perfect string match. While often too rigid, it serves as a useful baseline.

### Generating the Answer (`_generate_answer`)

This method simulates the "Generation" part of RAG. It takes the user's prompt and the retrieved context and asks the LLM to generate an answer. Notice the system prompt—it explicitly instructs the model to use **only** the provided context. This is a key instruction for testing `Faithfulness`.

```python
# From the SimpleRAGEvaluator class
def _generate_answer(self, input_prompt: str, context_str: str) -> str:
    """Generates an answer using the LLM based on the provided context."""
    self.logger.info("Generating answer based on context...")
    system_prompt = f"""You are a helpful assistant. Answer the user's question based *only* on the context provided.
Context:
{context_str}
"""
    # ... (API call to OpenAI) ...
```

### Running the Full Evaluation (`evaluate`)

This is the main orchestration method. It takes the user prompt, retrieved context, and a ground-truth answer, then performs the following steps:

1.  Calls `_generate_answer` to get the RAG system's output.
2.  Iterates through the list of metrics we want to run.
3.  Calls the appropriate evaluator for each metric.
4.  Catches any errors during evaluation.
5.  Structures and returns a final dictionary containing the generated output and all the scores.

## Putting It All Together: The Main Execution

The `if __name__ == "__main__":` block is where we run our demo.

**1. Define the Inputs:** We create sample data that mimics the output of a RAG system: a user prompt, the context documents retrieved by the retriever, and an "expected" or ground-truth answer.

```python
my_input_prompt = "What is the primary function of a MongoDB Atlas Search index?"

my_retrieved_context = [
    {"source": "doc1.txt", "content": "MongoDB Atlas Search allows you to create full-text search indexes on your data. It is built on Apache Lucene."},
    {"source": "doc2.txt", "content": "A search index in Atlas defines how your fields are indexed and searched. You can define analyzers and mappings for various data types."}
]

my_expected_answer = "It enables full-text search on data stored in MongoDB Atlas."
```

**2. Run the Evaluation:** We call `evaluator.evaluate()` with our data.

**3. View the Results:** Finally, we print the results in a clean JSON format.

## Analyzing the Results

Here’s what the output looks like:

```json
--- 📊 Final Evaluation Results ---
{
  "generated_output": "The primary function of a MongoDB Atlas Search index is to define how your fields are indexed and searched, allowing for full-text search capabilities on your data.",
  "evaluation_scores": {
    "Faithfulness": {
      "score": 1.0,
      "reason": ""
    },
    "Context Relevancy": {
      "score": 0.478125,
      "reason": ""
    },
    "Factuality": {
      "score": 0.6,
      "reason": ""
    },
    "Exact Match": {
      "score": 0,
      "reason": "Output does not exactly match the expected value."
    }
  }
}
```

Let's interpret these scores:

  * **`generated_output`**: The answer our system produced is detailed and seems correct.
  * **`Faithfulness: 1.0`**: Perfect score\! The LLM stuck strictly to the facts provided in the `my_retrieved_context`. This is a huge win.
  * **`Context Relevancy: 0.478`**: This score is mediocre. The evaluator believes that parts of the retrieved context were not fully relevant to answering the specific question. This might suggest our retriever could be more precise.
  * **`Factuality: 0.6`**: This score indicates that the answer is partially factual compared to the ground truth. The generated answer is more detailed, and this nuance is reflected here.
  * **`Exact Match: 0`**: As expected, the score is zero. The generated text is far more descriptive than our concise `my_expected_answer`. This highlights the limitation of exact matching and shows why semantic, AI-based metrics are so valuable.

### Conclusion and Next Steps

This simple script provides a powerful framework for RAG evaluation. By systematically measuring metrics like faithfulness and context relevancy, you can diagnose weaknesses in your pipeline, track improvements over time, and ultimately build more reliable and trustworthy AI applications.

From here, you can:

  * **Expand your test cases:** Evaluate your system against a larger, more diverse set of prompts and documents.
  * **Automate your evaluation:** Integrate this script into a CI/CD pipeline to automatically test changes to your RAG system.
  * **Experiment with different metrics:** Explore other evaluators offered by `autoevals` to gain deeper insights into your system's performance.

By making evaluation a core part of your development workflow, you can move from simply building a RAG system to building a *great* one.

-----

### The Complete Script (`demo.py`)

For your convenience, here is the full code to get you started.

```python
# demo.py
#
# A single-file script to demonstrate a simplified RAG evaluation.
# This version is structured like a more robust application, with
# internal client management, logging, and error handling.
#
# To run this script:
# 1. Make sure you have Python installed.
# 2. Install the required libraries:
#    pip install openai python-dotenv autoevals
# 3. Create a file named '.env' in the same directory and add your
#    Azure OpenAI credentials like this:
#    AZURE_OPENAI_ENDPOINT="your_endpoint_here"
#    AZURE_OPENAI_API_KEY="your_api_key_here"
# 4. Run the script from your terminal:
#    python demo.py

import os
import sys
import json
import logging
from typing import List, Dict, Any

# Third-party libraries
from dotenv import load_dotenv
from openai import AzureOpenAI # Or use: from openai import OpenAI
from autoevals import Factuality, init
from autoevals.ragas import ContextRelevancy, Faithfulness

# --- Class Definition ---

class SimpleRAGEvaluator:
    """
    A simplified, robust evaluator that first generates an answer based on context,
    and then scores that output against a series of metrics.
    """
    def __init__(self, generation_model: str = "gpt-4o-mini", evaluation_model: str = "gpt-4o-mini"):
        """
        Initializes the evaluator, sets up logging, and configures clients.

        Args:
            generation_model (str): The OpenAI model to use for generating answers.
            evaluation_model (str): The OpenAI model to use for scoring (evaluation).
        """
        # 1. Configure Logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # 2. Store configuration
        self.generation_model = generation_model
        self.evaluation_model = evaluation_model
        self.client = None

        # 3. Initialize clients and evaluators
        self._initialize_clients()
        self._initialize_evaluators()
        
        self.logger.info(f"✅ SimpleRAGEvaluator initialized. Generation: '{generation_model}', Evaluation: '{evaluation_model}'.")

    def _initialize_clients(self):
        """Loads environment variables and initializes the OpenAI client."""
        self.logger.info("Initializing OpenAI client...")
        load_dotenv()
        
        try:
            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            if not azure_endpoint or not api_key:
                raise ValueError("Azure endpoint or API key is not configured. Please set environment variables.")

            self.client = AzureOpenAI(
                api_version="2024-05-01-preview",
                azure_endpoint=azure_endpoint,
                api_key=api_key,
            )
            # Test connection
            self.client.models.list()
            self.logger.info("Successfully initialized and tested AzureOpenAI client.")
        except Exception as e:
            self.logger.error(f"Fatal Error: Could not initialize AzureOpenAI client. Check configuration. Details: {e}")
            sys.exit(1)

    def _initialize_evaluators(self):
        """Initializes the autoevals library and defines metrics."""
        self.logger.info("Initializing evaluation metrics...")
        # The init function sets a global client for all evaluators.
        # The 'model' argument is now passed to each evaluator individually.
        init(client=self.client)
        
        self.METRICS = {
            'Context Relevancy': ContextRelevancy(model=self.evaluation_model),
            'Faithfulness': Faithfulness(model=self.evaluation_model),
            'Factuality': Factuality(model=self.evaluation_model),
            'Exact Match': self._exact_match
        }
        self.logger.info("Core evaluation metrics are ready.")

    def _exact_match(self, output: str, expected: str, **kwargs) -> Dict[str, Any]:
        """A simple, internal exact match evaluator."""
        score = 1 if output.strip().lower() == expected.strip().lower() else 0
        return {
            'score': score,
            'reason': "Exact match." if score == 1 else "Output does not exactly match the expected value."
        }
        
    def _generate_answer(self, input_prompt: str, context_str: str) -> str:
        """Generates an answer using the LLM based on the provided context."""
        self.logger.info("Generating answer based on context...")
        system_prompt = f"""You are a helpful assistant. Answer the user's question based *only* on the context provided.
Context:
{context_str}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.generation_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {input_prompt}"}
                ],
                temperature=0.0
            )
            generated_output = response.choices[0].message.content.strip()
            self.logger.info(f"✨ Generated Answer: {generated_output}")
            return generated_output
        except Exception as e:
            self.logger.error(f"Failed to generate answer from LLM: {e}")
            return "[GENERATION FAILED]"


    def evaluate(
        self,
        input_prompt: str,
        retrieved_context: List[Dict],
        expected_output: str,
        metrics_to_run: List[str]
    ) -> Dict[str, Any]:
        """
        Generates an answer and runs a suite of evaluations.

        Args:
            input_prompt (str): The original question or input.
            retrieved_context (List[Dict]): Documents retrieved from the vector store.
            expected_output (str): The ground truth answer for scoring.
            metrics_to_run (List[str]): A list of metric names to run.

        Returns:
            A dictionary containing the generated output and evaluation scores.
        """
        # 1. Generate the answer
        context_str = "\n".join([json.dumps(doc) for doc in retrieved_context])
        generated_output = self._generate_answer(input_prompt, context_str)

        # 2. Run evaluations on the generated output
        scores = {}
        for metric_name in metrics_to_run:
            evaluator = self.METRICS.get(metric_name)
            if not evaluator:
                self.logger.warning(f"Metric '{metric_name}' not found. Skipping.")
                continue

            self.logger.info(f"Running evaluation for: {metric_name}...")
            
            try:
                if hasattr(evaluator, 'eval'):
                    result = evaluator.eval(input=input_prompt, output=generated_output, context=context_str, expected=expected_output)
                    scores[metric_name] = {'score': result.score, 'reason': getattr(result, 'reason', '')}
                else:
                    scores[metric_name] = evaluator(output=generated_output, expected=expected_output)
            except Exception as e:
                self.logger.error(f"Error evaluating metric '{metric_name}': {e}")
                scores[metric_name] = {'score': None, 'reason': f"Failed to run evaluation due to an error: {e}"}
        
        # 3. Structure and return the final results
        return {
            'generated_output': generated_output,
            'evaluation_scores': scores
        }


# --- Main Execution Block ---

if __name__ == "__main__":
    # 1. Instantiate the Evaluator
    # The class now handles its own client setup.
    evaluator = SimpleRAGEvaluator()

    # 2. Gather Your RAG System's Data
    my_input_prompt = "What is the primary function of a MongoDB Atlas Search index?"

    my_retrieved_context = [
        {"source": "doc1.txt", "content": "MongoDB Atlas Search allows you to create full-text search indexes on your data. It is built on Apache Lucene."},
        {"source": "doc2.txt", "content": "A search index in Atlas defines how your fields are indexed and searched. You can define analyzers and mappings for various data types."}
    ]

    my_expected_answer = "It enables full-text search on data stored in MongoDB Atlas."

    evaluator.logger.info("\n--- 📝 Inputs for Evaluation ---")
    evaluator.logger.info(f"Input Prompt: {my_input_prompt}")
    evaluator.logger.info(f"Expected Answer: {my_expected_answer}")
    evaluator.logger.info(f"Retrieved Context:\n{json.dumps(my_retrieved_context, indent=2)}")

    # 3. Run the Evaluation
    metrics_to_run = ['Faithfulness', 'Context Relevancy', 'Factuality', 'Exact Match']

    evaluation_results = evaluator.evaluate(
        input_prompt=my_input_prompt,
        retrieved_context=my_retrieved_context,
        expected_output=my_expected_answer,
        metrics_to_run=metrics_to_run
    )

    # 4. View the Results
    print("\n--- 📊 Final Evaluation Results ---")
    print(json.dumps(evaluation_results, indent=2))
    print("\n--- Demo Complete ---")
```
