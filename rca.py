"""
This script implements the Risk-Concealment Attacks (RCA) framework to evaluate the robustness of various Large Language Models (LLMs) against adversarial prompts in financial area.
"""
from openai import OpenAI, RateLimitError
import time
import httpx
from tqdm import tqdm
from utils import (
    is_refusal,
    format_as_text,
    save_experiment
)
import pandas as pd
import random
from prompts import (
    SYSTEM_PROMPT,
    STRUCTURED_R1_PROMPT,
    ADAPTIVE_FOLLOWUP_PROMPT,
    JUDGEMENT_PROMPT
)
import json
import logging
import argparse


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

auth_token: str = ""  # Add your auth token here
base_url: str = "" # Add your base URL here
client = OpenAI(api_key=auth_token,
                base_url=base_url,
                timeout=httpx.Timeout(30.0),
                max_retries=3,
            )


def generate_initial_prompt(
        harmful_query: str, 
        auxilia_model: str = "llama-3.1-70b", 
        temperature: float = 0.01) -> str:
    """
    Generate the initial prompt using the harmful query.
    Args:
        harmful_query (str): The harmful query to be used in the prompt.
        auxilia_model (str): The LLM model to be used for generating the prompt.
        temperature (float): The temperature for the model's response.
    Returns:
        str: The generated initial prompt.
    """
    user_prompt = STRUCTURED_R1_PROMPT.format(harmful_query=harmful_query)
    
    response = client.chat.completions.with_raw_response.create(
        model=auxilia_model,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=temperature
    )
    response = response.parse()

    return response.choices[0].message.content


def generate_follow_up_question(
        harmful_query: str, 
        conversation: list[dict], 
        round_number: int, 
        is_target_llm_refusal: bool, 
        model: str = "llama-3.1-70b", 
        temperature: float = 0.01) -> str:
    """
    Generate a follow-up question based on the conversation history, refusal signal and round number.
    Args:
        harmful_query (str): The original harmful query.
        conversation (list[dict]): The conversation history.
        round_number (int): The current round number.
        is_target_llm_refusal (bool): Whether the target LLM refused to answer in the last turn.
        model (str): The LLM model to be used for generating the prompt.
        temperature (float): The temperature for the model's response.
    Returns:
        str: The generated follow-up question.
    """
    followup_prompt: str = ADAPTIVE_FOLLOWUP_PROMPT.format(
                evil_prompt = harmful_query,
                dialogue_history = format_as_text(conversation),
                round_number = round_number,
                is_target_llm_refusal = is_target_llm_refusal
                )
    response = client.chat.completions.with_raw_response.create(
        model=model,
        messages=[{"role": "user", "content": followup_prompt}],
        temperature=temperature
    )
    
    return response.parse().choices[0].message.content


def call_target_llm(messages: list[dict], target_model: str, max_retries: int = 3, temperature: float = 0.01) -> str:
    """
    Call the target LLM with conversation messages.
    Args:
        messages (list[dict]): The conversation messages.
        target_model (str): The target LLM model to be called.
        max_retries (int): The maximum number of retries in case of rate limit errors.
        temperature (float): The temperature for the model's response.
    Returns:
        str: The response from the target LLM.
    """
    for _ in range(max_retries):
        try:
            if "o1" in target_model or "o3" in target_model:
                response = client.chat.completions.with_raw_response.create(
                    model=target_model,
                    messages=messages,
                )
            else:
                response = client.chat.completions.with_raw_response.create(
                    model=target_model,
                    messages=messages,
                    temperature=temperature
                )

            return response.parse().choices[0].message.content
        except RateLimitError as e:
            wait_time_string = e.response.headers.get("x-ratelimit-reset-tokens")
            wait_time = float(wait_time_string)
            logger.info(f"[RateLimit] waiting {wait_time:.1f}s")
            time.sleep(wait_time)


def assess_attack_success(
        dialogue_history: str, 
        judgement_llm: str = "gpt-4.1",
        temporature: float = 0.01, 
        max_retries: int = 3) -> str:
    """
    Call judgement LLM to assess if the attack is successful.
    Args:
        dialogue_history (str): The dialogue history of the conversation.
        judgement_llm (str): The judgement LLM model to be used.
        temporature (float): The temperature for the judgement LLM's response.
        max_retries (int): The maximum number of retries in case of rate limit errors.
    Returns:
        str: The judgement result from the judgement LLM.
    """
    for _ in range(max_retries):
        try:
            judgement_prompt = JUDGEMENT_PROMPT.format(dialogue_history = dialogue_history)
            response = client.chat.completions.with_raw_response.create(
                model=judgement_llm,
                messages=[{"role": "user", "content": judgement_prompt}],
                temperature=temporature
            )
            return response.parse().choices[0].message.content
        except RateLimitError as e:
            wait_time_string = e.response.headers.get("x-ratelimit-reset-tokens")
            wait_time = float(wait_time_string)
            logger.info(f"[RateLimit] waiting {wait_time:.1f}s")
            time.sleep(wait_time)


def run_rca_attack(
    harmful_query: str,
    max_num_rounds: int,
    target_model: str,
    auxilia_model: str = "llama-3.1-70b",
    judgement_model: str = "gpt-4.1",
    temperature: float = 0.01,
    max_retries: int = 3,
    short_responses: bool = True
) -> list[dict]:
    """
    Run a single RCA attack.
    Args:
        harmful_query (str): The harmful query to be used in the attack.
        max_num_rounds (int): The maximum number of rounds for the attack.
        target_model (str): The target LLM model to be attacked.
        auxilia_model (str): The auxiliary LLM model to be used for generating prompts.
        judgement_model (str): The judgement LLM model to be used for assessing attack success.
        temperature (float): The temperature for the models' responses.
        max_retries (int): The maximum number of retries in case of rate limit errors.
        short_responses (bool): Whether to ask for short responses from the target LLM.
    Returns:
        tuple: A tuple containing the judgement result, the conversation history, and the number of rounds taken.
    """
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    initial_prompt = generate_initial_prompt(harmful_query, auxilia_model, temperature)
    conversation.append({"role": "user", "content": initial_prompt})
    logger.info(f"Generated initial prompt: {initial_prompt}")

    initial_prompt_with_instruction = (
        initial_prompt + " Please keep your response concise and clear, no more than 3-4 sentences."
        if short_responses else initial_prompt
    )
    initial_response = call_target_llm([{"role": "user", "content": initial_prompt_with_instruction}], target_llm_model=target_model, max_retries=max_retries, temperature=temperature)
    conversation.append({"role": "assistant", "content": initial_response})
    logger.info(f"Initial Response: {initial_response}")

    for i in range(max_num_rounds):
        is_target_llm_refusal = False
        for j in range(max_retries):
            followup_prompt = generate_follow_up_question(
                harmful_query=harmful_query,
                conversation=conversation,
                round_number=i + 1,
                is_target_llm_refusal=is_target_llm_refusal
            )
            logger.info(f"Follow-up Prompt (Round {i+1}, Attempt {j+1}): {followup_prompt}")

            conversation.append({"role": "user", "content": followup_prompt + "Please keep your response concise and clear, no more than 3-4 sentences."})
            target_response = call_target_llm(conversation, target_model=target_model, temperature=temperature, max_retries=max_retries)
            logger.info(f"Target LLM Response (Round {i+1}, Attempt {j+1}): {target_response}")

            if is_refusal(target_response):
                is_target_llm_refusal = True
                continue
            else:
                conversation.append({"role": "assistant", "content": target_response})
                break
        
        dialogue_history=format_as_text(conversation)
        judgement_result=assess_attack_success(dialogue_history, judgement_llm=judgement_model, temporature=temperature, max_retries=max_retries)
        judgement_result=json.loads(judgement_result)
        if judgement_result["successful_attack"]:
            logger.info(f"Attack succeeded in round {i+1}")
            return judgement_result, conversation, i + 1

    return judgement_result, conversation, max_num_rounds


def run_rca_experiments(
        dataset_path: str, 
        target_llm_models: list[str],
        auxilia_model: str = "llama-3.1-70b",
        judgement_model: str = "gpt-4.1",
        max_num_rounds: int = 5,
        max_retries: int = 3,
        temperature: float = 0.01,
        do_sample: bool = True,
        sample_size: int = 80,
        ) -> None:
    """
    Run RCA experiments on a dataset of harmful queries.
    Args:
        dataset_path (str): The path to the dataset CSV file.
        target_llm_models (list[str]): A list of target LLM models to be attacked.
        auxilia_model (str): The auxiliary LLM model to be used for generating prompts.
        judgement_model (str): The judgement LLM model to be used for assessing attack success.
        max_num_rounds (int): The maximum number of rounds for each attack.
        max_retries (int): The maximum number of retries in case of rate limit errors.
        temperature (float): The temperature for the models' responses.
        do_sample (bool): Whether to sample a subset of the dataset for the experiments.
        sample_size (int): The number of samples to use if sampling is enabled.
    Returns:
        None
    """
    df = pd.read_csv(dataset_path)
    df = df[["data", "label"]]
    data_list = list(df.itertuples(index=False, name=None))

    if do_sample:
        data_list = random.sample(data_list, sample_size)

    # Initialize statistics
    model_stats = {
        model: {
            "success_count": 0,
            "total_count": 0,
            "time_total": 0.0,
            "experiment_data": []
        }
        for model in target_llm_models
    }

    # Iterate through data first
    for idx, data in enumerate(tqdm(data_list, desc="Running experiments"), 1):
        harmful_query, label = data
        for target_model in target_llm_models:
            stats = model_stats[target_model]
            start_time = time.perf_counter()
            try:
                # Run the attack
                judgement_result, conversation, num_round = run_rca_attack(
                    harmful_query=harmful_query,
                    max_num_rounds=max_num_rounds,
                    target_model=target_model,
                    auxilia_model=auxilia_model,
                    judgement_model=judgement_model,
                    temperature=temperature,
                    max_retries=max_retries,
                )
                dialogue_history = format_as_text(conversation)
                is_success = judgement_result["successful_attack"]
                stats["success_count"] += int(is_success)
                stats["total_count"] += 1

                stats["experiment_data"].append([
                    target_model, harmful_query, label,
                    is_success,
                    num_round,
                    judgement_result["reason"],
                    dialogue_history
                ])

            except Exception as e:
                logger.exception(f"Error for model {target_model} on data {idx}: {e}")

            end_time = time.perf_counter()
            stats["time_total"] += end_time - start_time

            # Show updated success rate
            success_rate = stats["success_count"] / stats["total_count"]
            logger.info(f"{target_model} | Success rate: {success_rate:.2%} ({stats['success_count']}/{stats['total_count']})")

    # Finalize: save per-model results and print summary
    for target_model, stats in model_stats.items():
        model_duration = round(stats["time_total"], 2)
        total = stats["total_count"]
        success = stats["success_count"]
        success_rate = success / total if total else 0

        save_experiment(stats["experiment_data"], target_model)

        logger.info(f"Model: {target_model} | Duration: {model_duration}s | Final success rate: {success_rate:.2%} ({success}/{total})")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run RCA experiments")

    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset file")
    parser.add_argument("--target_llm_models", type=str, nargs="+", required=True,
                        help="List of target LLM models (space separated)")
    parser.add_argument("--auxilia_model", type=str, default="llama-3.1-70b",
                        help="Auxiliary model name")
    parser.add_argument("--judgement_model", type=str, default="gpt-4.1",
                        help="Judgement model name")
    parser.add_argument("--max_num_rounds", type=int, default=5,
                        help="Maximum number of conversation rounds")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Maximum number of retries per round")
    parser.add_argument("--temperature", type=float, default=0.01,
                        help="Sampling temperature for LLMs")
    parser.add_argument("--do_sample", action="store_true",
                        help="Enable sampling instead of greedy decoding")
    parser.add_argument("--sample_size", type=int, default=80,
                        help="Number of samples to draw if sampling")

    args = parser.parse_args()

    run_rca_experiments(
        dataset_path=args.dataset_path,
        target_llm_models=args.target_llm_models,
        auxilia_model=args.auxilia_model,
        judgement_model=args.judgement_model,
        max_num_rounds=args.max_num_rounds,
        max_retries=args.max_retries,
        temperature=args.temperature,
        do_sample=args.do_sample,
        sample_size=args.sample_size,
    )