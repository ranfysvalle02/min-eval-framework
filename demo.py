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
from autoevals import Factuality, LLMClassifier, init
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


"""
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

--- Demo Complete ---

"""
